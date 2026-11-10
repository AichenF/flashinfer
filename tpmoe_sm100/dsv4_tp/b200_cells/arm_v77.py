"""Our DSV4-Flash TP4 arm: v77 (tpmoe_b2, B200 sm_100a) in routed-only mode (TPB2_SPEC=2: ids / weights given, no gate / shared),
per-M compile-time specialized builds (v77 dispatch flags + SPEC=2 + RS=0; M > 128: new exact builds with the M128 flags).
Weights = the SGLang / trtllm-gen tensors (sbtrt.prep_weights), used in place; numerics == trtllm-gen routed (bitwise)."""
import hashlib
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V77 = HERE / 'v77'
sys.path.insert(0, str(V77))
import tpb2            # noqa: E402  (v77 host: RankState, Weights, run)
import tpb2_dispatch   # noqa: E402

_EXT = {}

# routed-only extras on every build: SPEC=2 (no gate / shared at compile time), RS=0 (no in-kernel router code)
RO = ('-DTPB2_SPEC=2', '-DTPB2_RS=0', '-DTPB2_LLS=1')   # (+ kept: LLS sentinel all-reduce messages, bitwise; A/B with -DTPB2_LLS=0)
_LARGE = ('-DTPB2_ACTX=1', '-DTPB2_SFBI=1', '-DTPB2_ARV=34',   # (= the v77 M128 build's extras)
          '-DTPB2_XRLX=1')   # (suite_b200 kept: relaxed x-flag loads before the tables -- 16 serialized ld.acquire held the clear barrier ~4 us at M512)
LARGE_BUILDS = [(129, 255, _LARGE), (256, 256, _LARGE), (257, 383, _LARGE), (384, 384, _LARGE), (385, 511, _LARGE), (512, 512, _LARGE)]


def build_flags(M, extra=()):
    if M <= 128:
        f = tpb2_dispatch.build_flags(M)
    else:
        for lo, hi, ex in LARGE_BUILDS:
            if lo <= M <= hi:
                f = (f'-DTPB2_MMIN={lo}', f'-DTPB2_MMAX={hi}') + tpb2_dispatch.COMMON + ex
                break
    f = tuple(x for x in f if x.split('=')[0] not in ('-DTPB2_SPEC', '-DTPB2_RS'))
    if M >= 49:   # (suite_b200 kept: IDV -- vector id loads in build_tables; M128..512 -0.5..-3.2 us, bitwise)
        f = f + ('-DTPB2_IDV=1',)   # (A/B: IDV=2 below)
    ex = tuple(extra)
    over = {x.split('=')[0] for x in ex if x.startswith('-D')}
    return tuple(x for x in f + RO if x.split('=')[0] not in over) + ex


def load_ext(flags, src='tpmoe_b2.cu'):
    key = (src, flags)
    if key in _EXT:
        return _EXT[key]
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    tag = hashlib.md5((' '.join(flags) + src).encode()).hexdigest()[:12]
    build = Path(os.environ.get('SB_BUILD', str(HERE / 'build'))) / f'v77_{tag}'
    build.mkdir(parents=True, exist_ok=True)
    (build / 'flags.txt').write_text(' '.join(flags) + '\n')
    m = load(name=f'tpb2ro_{tag}', sources=[str(V77 / src)], build_directory=str(build),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK', *flags],
             extra_ldflags=['-lcuda'], verbose=False)
    _EXT[key] = m
    return m


class Arm:
    name = 'v77'

    def __init__(self, sh, rank, dev, gloo, ipc, trtW, Ms, extra=(), src='tpmoe_b2.cu'):
        import torch.distributed as dist
        assert sh.model == 'dsv4f' and sh.ndev in (4, 8), 'v77: DSV4-Flash TP4 / TP8'
        extra = tuple(extra) + (('-DTPB2_NDEV=8',) if sh.ndev == 8 else ())
        self.sh, self.rank, self.dev = sh, rank, dev
        self.flags = {M: build_flags(M, extra) for M in Ms}
        uniq = sorted(set(self.flags.values()))
        for i, f in enumerate(uniq):   # (parallel compiles: rank r builds every N-th distinct build first)
            if i % sh.ndev == rank:
                load_ext(f, src)
        dist.barrier(group=gloo)
        self.ext = {f: load_ext(f, src) for f in uniq}
        self.state, self.W = {}, {}
        self.keep = []
        for f, E in self.ext.items():
            st = tpb2.RankState(E, dev, sh.ndev, rank)
            if any(x.startswith('-DTPB2_MC=') and x != '-DTPB2_MC=0' for x in f):   # NVLS multicast pushes: symmetric buffers + multicast address
                import torch.distributed as dist
                import torch.distributed._symmetric_memory as symm_mem
                tabs = []
                for nb in (tpb2.RankState.RS_BYTES(E), tpb2.RankState.AG_BYTES(E)):
                    t = symm_mem.empty(nb, dtype=torch.uint8, device=dev)
                    t.zero_()
                    h = symm_mem.rendezvous(t, dist.group.WORLD.group_name)
                    self.keep.append((t, h))
                    off = t.data_ptr() - h.buffer_ptrs[rank]
                    assert h.multicast_ptr != 0, 'no NVLS multicast for the symmetric buffers'
                    tabs.append([h.buffer_ptrs[d] + off for d in range(sh.ndev)] + [h.multicast_ptr + off])
                torch.cuda.synchronize(); dist.barrier(group=gloo)
                st.link_ptrs(*tabs)
            else:
                pr, pa = ipc(tpb2.RankState.RS_BYTES(E)), ipc(tpb2.RankState.AG_BYTES(E))
                if int(getattr(E, 'LLS', 0)):   # (LLS) every all-reduce slot starts as the sentinel (readers re-arm their slots)
                    import ctypes
                    import torch.distributed as dist
                    import sbcommon as C
                    rt = C.CudaRT()
                    rt.ck(rt.lib.cudaMemset(ctypes.c_void_p(pr[rank]), 0xFF, int(tpb2.RankState.RS_BYTES(E))))
                    rt.ck(rt.lib.cudaMemset(ctypes.c_void_p(pa[rank]), 0xFF, int(tpb2.RankState.AG_BYTES(E))))
                    torch.cuda.synchronize(); dist.barrier(group=gloo)
                st.link_ptrs(pr, pa)
            self.state[f] = st
            self.W[f] = tpb2.Weights(E, trtW['w13'], trtW['s13'], trtW['w2'], trtW['s2'])

    def bind(self, M, x, ids, rw, out=None):
        f = self.flags[M]
        st, W = self.state[f], self.W[f]
        if out is None:
            out = torch.empty(M, self.sh.DIM, dtype=torch.bfloat16, device=self.dev)
        return (lambda: tpb2.run(st, W, x, out, ids=ids, rw=rw, factor=1.0)), out

    def run_dbg(self, M, x, ids, rw, out, dbg):   # (a -DTPB2_DBG=1 build: phase stamps into dbg [G][32])
        f = self.flags[M]
        st, W = self.state[f], self.W[f]
        st.dbg = dbg
        tpb2.run(st, W, x, out, ids=ids, rw=rw, factor=1.0)
        st.dbg = None

    @property
    def grid(self):
        return next(iter(self.state.values())).grid

