"""tp4v41 (H20 TP4 DeepSeek-V4.1-Flash, routed-only): suite_h20b src/tp4v41.cu = port_tp4v41.py on H20-TP8's tp8v41.cu (v41_h20 650c980;
h123 lineage v78p12dev) built -DTP2_MODEL=1 -DTP2_NDEV=4 -DTP2_SHARED=0; host = src/tp8v41.py (RankState / run); weights = the canonical
MXFP4 TP slices (ctx.tp_w()) in Humming's layout via src/v41_prep.py (main w13 stacks [512, 512], tail w13 [64, 64], w2 K 576).
--ours-so SO: a prebuilt extension instead of the build; --ours-opt src=FILE.cu (another source in src/, e.g. tp4v41b.cu)."""
import hashlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / '_h20b' / 'src'   # (tpmoe_suite import: suite_h20b/src @ ca18afa copied to kernels/_h20b/src)
sys.path.insert(0, str(SRC))
import tp8v41 as TV  # noqa: E402

ADAPTER_API = 1
FLAGS = ('-DTP2_MODEL=1', '-DTP2_NDEV=4', '-DTP2_SHARED=0')


class Cell:
    xq_mode = 'tok'

    def __init__(self, ctx):
        self.ctx, self.name = ctx, 'tp4v41'
        H, I, E, K = ctx.shape
        assert (H, I, E, K) == (5120, 2304, 384, 6) and ctx.ndev == 4, (ctx.shape, ctx.ndev)
        src = ctx.opts.get('src', 'tp4v41e.cu')
        if ctx.so:
            path = ctx.so[0]
        else:   # rank 0 builds into the node-local build root, then every rank loads the same build
            os.environ['TP_BUILD_ROOT'] = os.environ.get('SUITE_BUILD', '/tmp/tpsuite_build')
            tag = 'suite_' + Path(src).stem
            if ctx.rank == 0:
                TV.build(src, model=1, ndev=4, shared=0, tag=tag)
            dist.barrier(group=ctx.gloo)
            path = TV.build(src, model=1, ndev=4, shared=0, tag=tag).__file__
        self.E = TV.load_so(path)
        self.max_m = int(self.E.MAXM)
        self.st = TV.RankState(self.E, ctx.dev, alloc_comm=False)
        assert int(getattr(self.E, 'LL_PARITY', 0)) == 1, 'tp4v41e built without TP2_LL_PARITY (LL slots must be epoch-parity double-buffered)'
        self.st.link_ptrs(ctx.ipc_alloc(self.st.rs_bytes), ctx.ipc_alloc(self.st.ag_bytes))   # (exports hold both LL_PARITY halves)
        t0 = time.time()
        os.environ.setdefault('SGLANG_HUMMING_INPUT_QUANT_CONFIG', '{"dtype":"float8e4m3"}')
        import v41_prep
        w, _ = v41_prep.prep_rank('v41', 4, ctx.rank, ctx.dev, R=v41_prep.raw_from_canonical(ctx.tp_w()))
        self.w = TV.adapt_weights(self.E, {k: v.contiguous() for k, v in w.items()})
        ctx.log(f'  tp4v41: Humming weights {time.time() - t0:.0f} s, MAXM {self.max_m}, so {Path(path).name}')
        self.prov = dict(kernel=f'tp4v41 ({src}, port_tp4v41.py on tp8v41 650c980), routed-only, TP4', flags=list(FLAGS),
                         so={Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()},
                         src_sha256=hashlib.sha256((SRC / src).read_bytes()).hexdigest())

    def bind(self, M, x, ids, w, out):
        if M > self.max_m:
            return None
        E, st, wts, rank = self.E, self.st, self.w, self.ctx.rank

        def fn():
            TV.run(E, st, wts, ids, w, 1.0, out, rank, 4, x)
        return fn

    def pre_replay(self):   # the host launch tag (seq) is frozen inside a one-call graph: clear the per-token quant flags
        self.st.xflag.zero_()

    def info(self):
        return self.prov


def make_cell(ctx):
    return Cell(ctx)
