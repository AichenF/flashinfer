"""DSV4-Flash arm from the B300 tp8v41 lineage, built for sm_100a (B200): v4ft = suite_b300b/s3/port_v4f.py over tp8_v41_b300/tp8v41.cu
as committed at e066ec9 (v4ft/v4fti.cu, md5 84922b1c..., regenerated here bit-identical with `python3 v4ft/port_v4f.py rev:e066ec9`), host
v4ft/s3host.py (copied from suite_b300b/s3, S3_KSRC=v4fti).  Routed-only, x bf16 in (MXFP8 in the kernel), in-kernel one-shot / RS+AG LL
all-reduce (LLS sentinel messages + the LLS grid trap), NMAX 32 from M >= 256, GATHER 0 from M >= 256 (source default TPB2_G0_MINM).
Per-M exact builds (s3host.spec_flags(M, M, N) + per_m[M] + extra + ONESHOT_MAXM), one shared workspace (sized for the widest build).
F2H (H2, with T1PAR) per M via S3_F2H_MINM or -DTPB2_F2H=1 in the per-M flags; F2H + ACTX is refused by the source (static_assert)."""
import ctypes
import importlib
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def host(ndev):
    os.environ['S3_MODEL'] = 'v4f'
    os.environ['S3_TP'] = str(ndev)
    os.environ['S3_KSRC'] = os.environ.get('SB_V4FT_KSRC', 'v4fti')
    os.environ.setdefault('S3_BUILD', os.path.join(os.environ.get('SB_BUILD', str(HERE / 'build')), 'v4ft'))
    sys.path.insert(0, str(HERE / 'v4ft'))
    H = importlib.import_module('s3host')
    assert H.NDEV == ndev and H.MODEL == 'v4f', (H.NDEV, H.MODEL)
    return H


def parse_dispatch(s):
    """'384:-DA=1,-DB=2 512:-DA=1' -> {384: ('-DA=1', '-DB=2'), 512: ('-DA=1',)}"""
    d = {}
    for item in (s or '').split():
        m, _, fl = item.partition(':')
        d[int(m)] = tuple(f for f in fl.split(',') if f)
    return d


class Arm:
    kind = 'v4ft'

    def __init__(self, sh, rank, dev, gloo, ipc, raw, Ms, extra=(), oneshot_maxm=32, per_m=None):
        import torch.distributed as dist
        assert sh.model == 'dsv4f' and sh.ndev in (4, 8), 'DSV4-Flash TP4 / TP8'
        H = self.H = host(sh.ndev)
        assert (sh.DIM, sh.INTER_FULL, sh.NEXP) == (H.DIM, H.INTER_FULL, H.NEXP)
        self.name = 'v4ft'
        self.sh, self.rank, self.dev = sh, rank, dev
        per_m = per_m or {}
        self.fl = {M: H.spec_flags(M, M, sh.ndev, extra=tuple(per_m.get(M, ())) + tuple(extra) + (f'-DTPB2_ONESHOT_MAXM={oneshot_maxm}',)) for M in Ms}
        tag = lambda M: f'm{M}_' + ''.join(c for c in '_'.join(self.fl[M][4:]) if c.isalnum())[-40:]
        for i, M in enumerate(Ms):
            if i % sh.ndev == rank:
                H.load_ext(tag=tag(M), flags=self.fl[M])
        dist.barrier(group=gloo)
        self.ext = {M: H.load_ext(tag=tag(M), flags=self.fl[M]) for M in Ms}
        E0 = self.ext[Ms[0]]
        lls = {int(getattr(e, 'LLS', 0)) for e in self.ext.values()}
        assert len(lls) == 1, 'LLS and non-LLS builds must not share a workspace'
        self.raw = raw
        kl = H.kernel_layout(raw)
        self.f2h = any(int(getattr(e, 'F2H', 0)) for e in self.ext.values())
        if self.f2h:
            kl['f2h'] = H.f2h_weights(raw, rank)   # (collective)
        self.W = H.Weights(next((e for e in self.ext.values() if int(getattr(e, 'F2H', 0))), E0), kl)
        self.st = H.RankState(E0, dev, sh.ndev, rank)
        ptrs = ipc(max(H.RankState.RS_BYTES(e) for e in self.ext.values()))
        if lls.pop():   # (LLS) every all-reduce slot starts as the 0xFF sentinel (readers re-arm their slots)
            import sbcommon as C
            rt = C.CudaRT()
            rt.ck(rt.lib.cudaMemset(ctypes.c_void_p(ptrs[rank]), 0xFF, int(E0.RS_LL_BYTES)))
            torch.cuda.synchronize(); dist.barrier(group=gloo)
        self.st.link_ptrs(ptrs)
        if self.f2h:   # (H2) symmetric gathered-h buffers, zeroed (epoch ready tags start at 0), sized for the widest build
            self.st.hgp = ipc(max(int(getattr(e, 'HG_BYTES', 0)) for e in self.ext.values()))

    def is_f2h(self, M):
        return int(getattr(self.ext.get(M), 'F2H', 0))

    def ref_partial(self, x, ids, rw):   # (this arm's exact numerics; collective for F2H builds)
        if self.is_f2h(int(x.shape[0])):
            return self.H.ref_rank_h(self.raw, self.W.f2h_w2, self.W.f2h_s2, x, ids, rw, self.rank)
        return self.H.ref_rank(self.raw, x, ids, rw)

    def bind(self, M, x, ids, rw, out=None):
        if out is None:
            out = torch.empty(M, self.sh.DIM, dtype=torch.bfloat16, device=self.dev)
        E = self.ext[M]
        return (lambda: self.H.run(self.st, self.W, x, ids, rw, out, E=E)), out

    def run_dbg(self, M, x, ids, rw, out, dbg):
        self.H.run(self.st, self.W, x, ids, rw, out, dbg=dbg, E=self.ext[M])

    @property
    def grid(self):
        return self.st.grid
