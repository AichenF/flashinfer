"""tp41 (H20 TP8 DSV4.1-Flash, routed-only) for the tpmoe suite harness.  Kernel: tp8v41.cu (TP2_MODEL 1: H5120, I 288/rank, E384,
top-6) built from src/ (copied from the owner repo, commit in src/COMMIT) into $SUITE_BUILD by rank 0, or prebuilt via --ours-so.
Weights: this kernel's Humming layout (main 2 x 128-col items + the T4 split tail + w2 9 k32 slices) from the canonical MXFP4 TP
slices (ctx.tp_w()) via src/v41_prep.humming_from_raw (humming-kernels 0.1.10).  RS / AG LL buffers: ctx.ipc_alloc.
Numerics: per-token E4M3 x (xq_mode tok), in-kernel LL all-reduce; pre_replay clears xflag (host seq tag frozen in a one-call graph)."""
import hashlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
SRC = HERE / 'src'
ADAPTER_API = 1


def _load_so(path):
    import importlib.machinery
    import importlib.util
    name = Path(path).stem
    loader = importlib.machinery.ExtensionFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


class Tp41Cell:
    xq_mode = 'tok'
    name = 'tp41'

    def __init__(self, ctx):
        self.ctx = ctx
        H, I, E, K = ctx.shape
        assert (H, I, E, K) == (5120, 2304, 384, 6) and ctx.ndev == 8 and ctx.sm == 90, 'tp41: DSV4.1-Flash TP8 on SM90 only'
        sys.path.insert(0, str(SRC))
        import tp8v41 as TV
        self.TV = TV
        if ctx.so:
            path = ctx.so[0].partition('@')[0]
            self.E = _load_so(path)
            self.so_path = path
        else:
            root = os.environ.get('SUITE_BUILD', '/tmp/tpsuite_build')
            flags = [f for f in ctx.opts.get('flags', '').split(':') if f]

            def build():
                from torch.utils.cpp_extension import load
                tag = 'tp8v41_m1_n8_s0' + ''.join('_' + ''.join(c if c.isalnum() else '_' for c in f) for f in flags)
                bdir = Path(root) / tag
                bdir.mkdir(parents=True, exist_ok=True)
                os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
                return load(name=tag, sources=[str(SRC / 'tp8v41.cu')], build_directory=str(bdir),
                            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-I{SRC}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
                                               '-DTP2_MODEL=1', '-DTP2_NDEV=8', '-DTP2_SHARED=0', *flags], verbose=False)
            if ctx.rank == 0:
                build()
            dist.barrier(group=ctx.gloo)
            self.E = build()
            self.so_path = self.E.__file__
            # M-dispatch (8-GPU A/B 11-09): the T4N16 build (8 KB x tiles, NST 6) for M 9..16 and M >= 320, the NST-7 build
            # (-DTP2_T4N16=0) elsewhere
            if not ctx.opts.get('flags') and ctx.opts.get('dispatch', '1') != '0':
                flags.append('-DTP2_T4N16=0')
                if ctx.rank == 0:
                    build()
                dist.barrier(group=ctx.gloo)
                self.E_n7 = build()
        E0 = self.E
        assert int(E0.MODEL) == 1 and int(E0.NDEV) == 8
        self.max_m = int(E0.MAXM)
        self.st = TV.RankState(E0, ctx.dev, alloc_comm=False)
        self.st.link_ptrs(ctx.ipc_alloc(self.st.rs_bytes), ctx.ipc_alloc(self.st.ag_bytes))
        t0 = time.time()
        from v41_prep import humming_from_raw
        tw = ctx.tp_w()
        i = I // ctx.ndev
        R = dict(g=tw['w13'][:, :i], gs=tw['s13'][:, :i], u=tw['w13'][:, i:], us=tw['s13'][:, i:], w2=tw['w2'], w2s=tw['s2'])
        R = {k: v.to(ctx.dev).contiguous() for k, v in R.items()}
        self.w = TV.adapt_weights(E0, {k: v.contiguous() for k, v in humming_from_raw(R, H, i, E, ctx.dev).items()})
        torch.cuda.synchronize()
        ctx.log(f'  tp41: Humming weights {time.time() - t0:.0f} s, MAXM {self.max_m}, so {self.so_path}')

    def ext_of(self, M):
        if getattr(self, 'E_n7', None) is not None and not (9 <= M <= 16 or M >= 320):
            return self.E_n7
        return self.E

    def bind(self, M, x, ids, w, out):
        if M > self.max_m:
            return None
        TV, E0, st, wts, rank = self.TV, self.ext_of(M), self.st, self.w, self.ctx.rank

        def fn():
            TV.run(E0, st, wts, ids, w, 1.0, out, rank, 8, x)
        return fn

    def pre_replay(self):
        self.st.xflag.zero_()

    def info(self):
        commit = (SRC / 'COMMIT').read_text().strip() if (SRC / 'COMMIT').exists() else '?'
        return dict(kernel='tp41 (tp8v41.cu TP2_MODEL 1, routed-only, TP8)', commit=commit, so=self.so_path,
                    dispatch='T4N16 build at M 9..16 and >= 320, -DTP2_T4N16=0 build elsewhere' if getattr(self, 'E_n7', None) is not None else 'single',
                    so_sha256=hashlib.sha256(Path(self.so_path).read_bytes()).hexdigest(), flags=self.ctx.opts.get('flags', ''))


def make_cell(ctx):
    return Tp41Cell(ctx)
