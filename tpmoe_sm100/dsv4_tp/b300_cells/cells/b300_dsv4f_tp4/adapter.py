"""v4ft (agent B300b) as the suite's `ours` arm for cell b300_dsv4f_tp4: B300 TP-MoE kernel v4ft.cu (port_v4f.py from tp8_v41_b300/tp8v41.cu, the B300-TP8 lineage) + s3host.py,
copied from /home/scratch.aichenf_wwfo/suite_b300b/s3 @ bba04fa (see README.md).  One compile-time specialized build per M
(s3host.spec_flags(M, M, 4) + the cell's per-M flags below), compiled in parallel over the ranks into $SUITE_BUILD/b300_dsv4f_tp4.
Weights: s3host.kernel_layout of the canonical MXFP4 TP slices (ctx.tp_w()).  Exact numerics reference: s3host.ref_rank.
LLS builds (v4ft default): the RS buffer's LL region is 0xFF-filled once (sentinel slots, re-armed by their readers); LLS and non-LLS
builds never share a workspace.  --opt flags="-DX=.." adds build flags to every M; oneshot=N sets ONESHOT_MAXM (default 64)."""
import ctypes
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
os.environ['S3_MODEL'] = 'v4f'
os.environ['S3_TP'] = '4'
os.environ['S3_KSRC'] = 'v4ft'
os.environ['S3_BUILD'] = os.path.join(os.environ.get('SUITE_BUILD', '/tmp/tpsuite_build'), 'b300_dsv4f_tp4')
sys.path.insert(0, str(HERE))
import s3host as H  # noqa: E402

ADAPTER_API = 1
COMMIT = 'bba04fa'
ONESHOT = 64
PER_M = {384: ('-DTPB2_ACTX=1', '-DTPB2_T1PW=2'), 512: ('-DTPB2_ACTX=1', '-DTPB2_T1PW=2')}   # M -> extra build flags (the cell's kept configuration)


class Cell:
    xq_mode = 'mx32b'
    max_m = 512

    def __init__(self, ctx):
        assert ctx.model == 'dsv4f' and ctx.ndev == 4, 'v4ft b300_dsv4f_tp4: dsv4f TP4 only'
        assert tuple(ctx.shape[:3]) == (H.DIM, H.INTER_FULL, H.NEXP), (ctx.shape, H.DIM, H.INTER_FULL, H.NEXP)
        self.ctx = ctx
        self.name = f'v4ft@{COMMIT}'
        self.oneshot = int(ctx.opts.get('oneshot', ONESHOT))
        self.extra = tuple(ctx.opts.get('flags', '').split())

    def flags(self, M):
        ex = PER_M.get(M, ()) + self.extra + (f'-DTPB2_ONESHOT_MAXM={self.oneshot}',)
        return H.spec_flags(M, M, 4, extra=ex)

    def tag(self, M):
        return f'm{M}_' + ''.join(c for c in '_'.join(self.flags(M)[4:]) if c.isalnum())[-40:]

    def prepare(self, Ms):
        ctx = self.ctx
        t0 = time.time()
        for i, M in enumerate(Ms):
            if i % ctx.ndev == ctx.rank:
                H.load_ext(tag=self.tag(M), flags=self.flags(M))
        dist.barrier(group=ctx.gloo)
        self.ext = {M: H.load_ext(tag=self.tag(M), flags=self.flags(M)) for M in Ms}
        E0 = self.ext[Ms[0]]
        lls = {int(getattr(e, 'LLS', 0)) for e in self.ext.values()}
        assert len(lls) == 1, 'LLS and non-LLS builds must not share a workspace'
        self.raw = ctx.tp_w()
        kl = H.kernel_layout(self.raw)
        self.f2h = any(int(getattr(e, 'F2H', 0)) for e in self.ext.values())   # ((v41 H2) per-M: FC2 sharded on H at large M)
        if self.f2h:
            kl['f2h'] = H.f2h_weights(self.raw, ctx.rank)   # (collective)
        self.W = H.Weights(next((e for e in self.ext.values() if int(getattr(e, 'F2H', 0))), E0), kl)
        self.st = H.RankState(E0, ctx.dev, 4, ctx.rank)
        ptrs = ctx.ipc_alloc(max(H.RankState.RS_BYTES(e) for e in self.ext.values()))
        if lls.pop():   # (LLS: every all-reduce slot starts as the 0xFF sentinel; readers re-arm their slots)
            import tps_common as C
            C._RT.ck(C._RT.lib.cudaMemset(ctypes.c_void_p(ptrs[ctx.rank]), 0xFF, int(E0.RS_LL_BYTES)))
            torch.cuda.synchronize()
            dist.barrier(group=ctx.gloo)
        self.st.link_ptrs(ptrs)
        if self.f2h:   # ((v41 H2) the symmetric gathered-h buffers, zeroed by ipc_alloc; sized for the widest build)
            self.st.hgp = ctx.ipc_alloc(max(int(getattr(e, 'HG_BYTES', 0)) for e in self.ext.values()))
        ctx.log(f'  v4ft b300_dsv4f_tp4: {len(Ms)} builds + weights {time.time() - t0:.0f} s (smem {E0.SMEM} B, LLS {int(getattr(E0, "LLS", 0))})')

    def bind(self, M, x, ids, w, out):
        if M not in self.ext:
            return None
        E, st, W = self.ext[M], self.st, self.W
        return lambda: H.run(st, W, x, ids, w, out, E=E)

    def ref_partial(self, x, ids, w):
        E = self.ext.get(int(x.shape[0]))
        if E is not None and int(getattr(E, 'F2H', 0)):   # ((v41 H2) the H2 numerics: full-K FC2 per row block (collective: all ranks call it))
            return H.ref_rank_h(self.raw, self.W.f2h_w2, self.W.f2h_s2, x, ids, w, self.ctx.rank)
        return H.ref_rank(self.raw, x, ids, w)

    def info(self):
        return dict(kernel=f'v4ft (suite_b300b/s3 @ {COMMIT}), cell b300_dsv4f_tp4, per-M spec builds', oneshot=self.oneshot, extra=list(self.extra),
                    per_m={str(k): list(v) for k, v in PER_M.items()},
                    flags={M: list(self.flags(M)) for M in getattr(self, 'ext', {})},
                    builds={M: getattr(e, '__file__', '?') for M, e in getattr(self, 'ext', {}).items()})


def make_cell(ctx):
    return Cell(ctx)
