"""tp8v41 LIVE (B300 TP8 DSV4.1-Flash routed-only): the tp8_v41_b300 root sources (set TP8_COMMIT for provenance); one compile-time
specialized build per M (v41host.spec_flags(M, M, 8) + -DTPB2_ONESHOT_MAXM=32, as x021 / x108; GATHER=0 for M >= 384 is the .cu
default), builds compiled in parallel over the ranks into $SUITE_BUILD.  Weights: v41host.kernel_layout of the canonical MXFP4 TP
slices (gate rows 288 r.. | up rows ..; w2 K columns 288 r..).  Exact numerics reference: v41host.ref_rank (bitwise target).
--opt flags="-DX=.." adds build flags; oneshot=N changes ONESHOT_MAXM."""
import os
import sys
import time
from pathlib import Path

import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent   # (the tp8_v41_b300 root: live tp8v41.cu / v41host.py)
sys.path.insert(0, str(HERE))
os.environ['TP8_BUILD'] = os.path.join(os.environ.get('SUITE_BUILD', '/tmp/tpsuite_build'), 'b300_dsv41f_tp8')
import v41host as H  # noqa: E402

ADAPTER_API = 1


class Cell:
    name = 'tp8v41@' + os.environ.get('TP8_COMMIT', 'live')
    xq_mode = 'mx32b'
    max_m = 512

    def __init__(self, ctx):
        assert ctx.model == 'dsv41f' and ctx.ndev == 8, 'tp8v41: DSV4.1-Flash TP8 only'
        self.ctx = ctx
        self.extra = tuple(ctx.opts.get('flags', '').split()) + (f'-DTPB2_ONESHOT_MAXM={ctx.opts.get("oneshot", 32)}',)

    def tag(self, M):
        return f'm{M}_' + ''.join(c for c in '_'.join(self.extra) if c.isalnum())[:40]

    def prepare(self, Ms):
        ctx = self.ctx
        t0 = time.time()
        fl = lambda M: H.spec_flags(M, M, 8, extra=self.extra)  # noqa: E731
        for i, M in enumerate(Ms):
            if i % ctx.ndev == ctx.rank:
                H.load_ext(tag=self.tag(M), flags=fl(M))
        dist.barrier(group=ctx.gloo)
        self.ext = {M: H.load_ext(tag=self.tag(M), flags=fl(M)) for M in Ms}
        E0 = self.ext[Ms[0]]
        self.raw = ctx.tp_w()
        kl = H.kernel_layout(self.raw)
        self.f2h = any(int(getattr(e, 'F2H', 0)) for e in self.ext.values())   # ((v41 H2) per-M: FC2 sharded on H at large M)
        if self.f2h:
            kl['f2h'] = H.f2h_weights(self.raw, ctx.rank)
        self.W = H.Weights(next((e for e in self.ext.values() if int(getattr(e, 'F2H', 0))), E0), kl)
        self.st = H.RankState(E0, ctx.dev, 8, ctx.rank)
        ptrs = ctx.ipc_alloc(H.RankState.RS_BYTES(E0))
        if int(getattr(E0, 'LLS', 0)):   # (LLS builds: every all-reduce slot starts as the 0xFF sentinel; readers re-arm their slots)
            import ctypes
            import tps_common as C
            C._RT.ck(C._RT.lib.cudaMemset(ctypes.c_void_p(ptrs[ctx.rank]), 0xFF, int(E0.RS_LL_BYTES)))
            C._RT.ck(C._RT.lib.cudaDeviceSynchronize())
            dist.barrier(group=ctx.gloo)
        self.st.link_ptrs(ptrs)
        if self.f2h:   # ((v41 H2) the symmetric gathered-h buffers, zeroed by ipc_alloc (the epoch ready tags must start at 0); one fixed layout for every build
            self.st.hgp = ctx.ipc_alloc(max(int(getattr(e, 'HG_BYTES', 0)) for e in self.ext.values()))
        ctx.log(f'  tp8v41: {len(Ms)} builds + weights {time.time() - t0:.0f} s (smem {E0.SMEM} B)')

    def bind(self, M, x, ids, w, out):
        E, st, W = self.ext[M], self.st, self.W
        return lambda: H.run(st, W, x, ids, w, out, E=E)

    def ref_partial(self, x, ids, w):
        E = self.ext.get(int(x.shape[0]))
        if E is not None and int(getattr(E, 'F2H', 0)):   # ((v41 H2) the H2 numerics: full-K FC2 per 640-row slice (collective: all ranks call it))
            return H.ref_rank_h(self.raw, self.W.f2h_w2, self.W.f2h_s2, x, ids, w, self.ctx.rank)
        return H.ref_rank(self.raw, x, ids, w)

    def info(self):
        return dict(kernel='tp8v41 @ ' + os.environ.get('TP8_COMMIT', 'live') + ' (tp8_v41_b300 root; per-M defaults: GATHER=0 M>=256, NMAX 32 M>=256, H2 (F2H) M>=128)', flags=list(self.extra),
                    builds={M: getattr(e, '__file__', '?') for M, e in getattr(self, 'ext', {}).items()})


def make_cell(ctx):
    return Cell(ctx)
