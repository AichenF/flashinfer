"""v41tp4 (DSV4.1-Flash TP4) as the suite's `ours` arm for cell b300_dsv41f_tp4 (agent B300b): the B200 agent's kernel
(suite_b200 @ 7eaa18a: v41tp4/v41tp4t.cu = port_tp4.py output of tp8v41.cu -- 9 full FC1 tiles, F2T tail-first FC2 with K64 tails, NDEV 4 --
plus its tail stage patch_tail.py: LLS sentinel all-reduce, TAILB guarded, gridDim trap) copied unmodified into suite_b300b/v41tp4 and
compiled for sm_103a.  LLS builds: the RS buffer's LL region is 0xFF-filled once; LLS and non-LLS builds never share a workspace.  One compile-time specialized build per M (v41tp4host.spec_flags(M, M, 4)
+ -DTPB2_ONESHOT_MAXM=64 + per-M flags), compiled in parallel over the ranks into $SUITE_BUILD/b300_dsv41f_tp4.
Weights: v41tp4host.kernel_layout of the canonical MXFP4 TP slices.  Exact numerics reference: v41tp4host.ref_rank.
--opt flags="-DX=.." adds build flags; oneshot=N sets ONESHOT_MAXM."""
import os
import sys
import time
from pathlib import Path

import ctypes

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
os.environ['TP8_BUILD'] = os.path.join(os.environ.get('SUITE_BUILD', '/tmp/tpsuite_build'), 'b300_dsv41f_tp4')
sys.path.insert(0, str(HERE))
import v41tp4host as H  # noqa: E402

ADAPTER_API = 1
COMMIT = 'f8b6cec'
ONESHOT = 64
PER_M = {384: ('-DTPB2_T1PW=2',), 512: ('-DTPB2_T1PW=2',)}
SRC = 'v41tp4t.cu'
COMMON = ('-DTPB2_LLS=1',)   # (every build: e.g. -DTPB2_LLS=1)


class Cell:
    xq_mode = 'mx32b'
    max_m = 512

    def __init__(self, ctx):
        assert ctx.model == 'dsv41f' and ctx.ndev == 4, 'v41tp4: DSV4.1-Flash TP4 only'
        self.ctx = ctx
        self.name = f'v41tp4@{COMMIT}'
        self.oneshot = int(ctx.opts.get('oneshot', ONESHOT))
        self.extra = tuple(ctx.opts.get('flags', '').split())

    def flags(self, M):
        return H.spec_flags(M, M, 4, extra=COMMON + PER_M.get(M, ()) + self.extra + (f'-DTPB2_ONESHOT_MAXM={self.oneshot}',))

    def tag(self, M):
        return f'm{M}_' + ''.join(c for c in '_'.join(self.flags(M)[4:]) if c.isalnum())[-40:]

    def prepare(self, Ms):
        ctx = self.ctx
        t0 = time.time()
        for i, M in enumerate(Ms):
            if i % ctx.ndev == ctx.rank:
                H.load_ext(src=SRC, tag=self.tag(M), flags=self.flags(M))
        dist.barrier(group=ctx.gloo)
        self.ext = {M: H.load_ext(src=SRC, tag=self.tag(M), flags=self.flags(M)) for M in Ms}
        E0 = self.ext[Ms[0]]
        self.raw = ctx.tp_w()
        self.W = H.Weights(E0, H.kernel_layout(self.raw))
        self.st = H.RankState(E0, ctx.dev, 4, ctx.rank)
        lls = {int(getattr(e, 'LLS', 0)) for e in self.ext.values()}
        assert len(lls) == 1, 'LLS and non-LLS builds must not share a workspace'
        ptrs = ctx.ipc_alloc(max(H.RankState.RS_BYTES(e) + int(getattr(e, 'TB_BYTES', 0)) for e in self.ext.values()))   # (+ the tail stage's TAILB area)
        if lls.pop():   # (LLS: every all-reduce slot starts as the 0xFF sentinel; readers re-arm their slots)
            import tps_common as C
            C._RT.ck(C._RT.lib.cudaMemset(ctypes.c_void_p(ptrs[ctx.rank]), 0xFF, int(E0.RS_LL_BYTES)))
            torch.cuda.synchronize()
            dist.barrier(group=ctx.gloo)
        self.st.link_ptrs(ptrs)
        ctx.log(f'  v41tp4 b300: {len(Ms)} builds + weights {time.time() - t0:.0f} s (smem {E0.SMEM} B, F2T {getattr(E0, "F2T", "?")}, LLS {int(getattr(E0, "LLS", 0))})')

    def bind(self, M, x, ids, w, out):
        if M not in self.ext:
            return None
        E, st, W = self.ext[M], self.st, self.W
        return lambda: H.run(st, W, x, ids, w, out, E=E)

    def ref_partial(self, x, ids, w):
        return H.ref_rank(self.raw, x, ids, w)

    def info(self):
        return dict(kernel=f'v41tp4 ({SRC} of suite_b200 @ 7eaa18a, copied to suite_b300b/v41tp4 @ {COMMIT}), per-M spec builds, sm_103a',
                    flags={M: list(self.flags(M)) for M in getattr(self, 'ext', {})},
                    oneshot=self.oneshot, extra=list(self.extra), per_m={str(k): list(v) for k, v in PER_M.items()},
                    builds={M: getattr(e, '__file__', '?') for M, e in getattr(self, 'ext', {}).items()})


def make_cell(ctx):
    return Cell(ctx)
