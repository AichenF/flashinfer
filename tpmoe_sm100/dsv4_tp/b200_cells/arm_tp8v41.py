"""Our DSV4.1-Flash arms from the B300 lineage, built for sm_100a (B200):
  TP8: tp8v41 (port_v41.py of k3h.cu; routed-only, in-kernel one-shot / RS+AG all-reduce).  Default base (11-09): tp8v41h2/ = tp8_v41_b300
       HEAD (d91f622: LLS, NMAX 32 at M >= 256, H2 / F2H behind TP8_F2H_MINM) + the LLS grid trap (tp8v41g.cu); the earlier base
       (ee6b585 + patch_tail) with SB_TP8V41_DIR=tp8v41 SB_V41SRC=tp8v41t.cu
  TP4: v41tp4 (port_tp4.py of tp8v41.cu: 9 full FC1 tiles, ragged K3 FC2 path for K 576, NDEV 4), v41tp4/
Per-M exact builds (host.spec_flags(M, M, N) + ONESHOT_MAXM), one shared workspace."""
import importlib
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def host(ndev):
    d, m = {8: (os.environ.get('SB_TP8V41_DIR', 'tp8v41h2'), 'v41host'), 4: (os.environ.get('SB_V41TP4_DIR', 'v41tp4'), 'v41tp4host')}[ndev]   # (TP8 base: tp8_v41_b300 HEAD, 11-09)
    sys.path.insert(0, str(HERE / d))
    return importlib.import_module(m)


class Arm:
    def __init__(self, sh, rank, dev, gloo, ipc, raw, Ms, extra=(), oneshot_maxm=32, src=None):
        import torch.distributed as dist
        assert sh.model == 'dsv41' and sh.ndev in (4, 8), 'DSV4.1-Flash TP4 / TP8'
        os.environ.setdefault('TP8_BUILD', os.environ.get('SB_BUILD', str(HERE / 'build')))
        H = self.H = host(sh.ndev)
        self.name = 'tp8v41' if sh.ndev == 8 else 'v41tp4'
        tp8dir = os.environ.get('SB_TP8V41_DIR', 'tp8v41h2')
        src = src or os.environ.get('SB_V41SRC') or (('tp8v41g.cu' if tp8dir == 'tp8v41h2' else 'tp8v41t.cu') if sh.ndev == 8 else 'v41tp4t.cu')
        self.base = tp8dir if sh.ndev == 8 else os.environ.get('SB_V41TP4_DIR', 'v41tp4')
        self.sh, self.rank, self.dev = sh, rank, dev
        ex = tuple(extra) + (f'-DTPB2_ONESHOT_MAXM={oneshot_maxm}',)
        if not any(f.startswith('-DTPB2_LLS=') for f in ex):
            ex = ('-DTPB2_LLS=1',) + ex   # (kept: LLS sentinel all-reduce messages; pass -DTPB2_LLS=0 for the LL tail)
        self.fl = {M: H.spec_flags(M, M, sh.ndev, extra=ex) for M in Ms}
        tag = lambda M: f'm{M}_n{sh.ndev}_' + ''.join(c for c in '_'.join(ex) if c.isalnum())[:40]
        for i, M in enumerate(Ms):
            if i % sh.ndev == rank:
                H.load_ext(src=src, tag=tag(M), flags=self.fl[M])
        dist.barrier(group=gloo)
        self.ext = {M: H.load_ext(src=src, tag=tag(M), flags=self.fl[M]) for M in Ms}
        E0 = self.ext[Ms[0]]
        assert len({int(getattr(e, 'LLS', 0)) for e in self.ext.values()}) == 1, 'LLS and tagged-LL builds must not share a workspace'
        kl = H.kernel_layout(raw)
        self.f2h = any(int(getattr(e, 'F2H', 0)) for e in self.ext.values())   # ((v41 H2, tp8v41h2/) FC2 sharded on H: all-gathered H-sharded W2)
        if self.f2h:
            kl['f2h'] = H.f2h_weights(raw, rank)   # (collective)
        self.W = H.Weights(next((e for e in self.ext.values() if int(getattr(e, 'F2H', 0))), E0), kl)
        self.st = H.RankState(E0, dev, sh.ndev, rank)
        ptrs = ipc(H.RankState.RS_BYTES(E0) + int(getattr(E0, 'TB_BYTES', 0)))   # (+ the TAILB areas; the H2 host's RS_BYTES already counts them: harmless extra)
        if int(getattr(E0, 'LLS', 0)):   # (LLS) every all-reduce slot starts as the sentinel (readers re-arm their slots)
            import ctypes
            import sbcommon as C
            rt = C.CudaRT()
            rt.ck(rt.lib.cudaMemset(ctypes.c_void_p(ptrs[rank]), 0xFF, int(E0.RS_LL_BYTES)))
            torch.cuda.synchronize(); dist.barrier(group=gloo)
        self.st.link_ptrs(ptrs)
        if self.f2h:   # ((v41 H2) the symmetric gathered-h buffers, zeroed (the epoch ready tags start at 0); one layout for every build
            self.st.hgp = ipc(max(int(getattr(e, 'HG_BYTES', 0)) for e in self.ext.values()))

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
