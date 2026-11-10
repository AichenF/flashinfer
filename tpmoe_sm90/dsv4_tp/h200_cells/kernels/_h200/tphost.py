"""Host side for every tpmoe build used on H200 (one wrapper): v94ex / h123-family sources (tp4moe2_*.cu, old run ABI) and the
tp8v41-family sources (port_v41.py output, run ABI + tail weights).  Extension passed explicitly (several builds per process).
Derived from v41_h20/tp8v41.py (H20-TP8 agent); changes: build() for any source / NDEV / MODEL, RS/AG sized for the epoch-parity
double buffers (v94par/v94ex: 2x) whatever the build, ABI auto-detected (tail-weight builds export NSL2)."""
import importlib.machinery
import importlib.util
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE / 'src'
BUILDS = Path(os.environ.get('TP_BUILDS', str(HERE / 'builds')))   # (node-local: scratch is full)


def build(src, ndev, model=None, shared=0, extra=(), tag=None, verbose=False):
    """build (or load the cached build of) SRC/src with -DTP2_NDEV [-DTP2_MODEL] -DTP2_SHARED; returns the module"""
    from torch.utils.cpp_extension import load
    flags = [f'-DTP2_NDEV={ndev}', f'-DTP2_SHARED={shared}'] + ([f'-DTP2_MODEL={model}'] if model is not None else []) + list(extra)
    stem = Path(src).stem.replace('.', '_')
    tag = tag or (f'n{ndev}_s{shared}' + (f'_m{model}' if model is not None else '')
                  + ''.join('_' + ''.join(c if c.isalnum() else '_' for c in f) for f in extra))
    name = f'{stem}_{tag}'
    bdir = BUILDS / name
    bdir.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    return load(name=name, sources=[str(SRC / src)], build_directory=str(bdir),
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-I{SRC}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK', *flags],
                verbose=verbose)


def load_so(path):
    path = str(path)
    name = Path(path).stem
    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def comm_bytes(E):
    base = 2 * E.NRT * E.MAXM * E.NMSG * 16   # (v94par / v94ex: two epoch-parity halves; older builds use the first half)
    return max(int(getattr(E, 'RS_BYTES', 0)), base), max(int(getattr(E, 'AG_BYTES', 0)), base)


class RankState:
    """Per-rank workspace for extension E (sized for the largest of the builds that share it: pass the biggest E)."""

    def __init__(self, E, device, alloc_comm=True):
        self.E = E
        dim = getattr(E, 'DIM', 4096)
        inter, nqg, nkt2 = E.INTER, E.NQG, E.NKT2
        nexp = getattr(E, 'NEXP', 256)
        self.device = device
        grid = torch.cuda.get_device_properties(device).multi_processor_count
        z = dict(device=device)
        self.act = torch.zeros(E.MAXE * inter, dtype=torch.bfloat16, **z)
        self.amax = torch.zeros(E.MAXE * nqg, dtype=torch.float32, **z)
        self.part = torch.zeros(2 * grid * E.NCONS * 32, dtype=torch.float32, **z)
        self.act_flags = torch.zeros(E.MAXP * 4, dtype=torch.int32, **z)
        self.part_flags = torch.zeros(max(2 * grid, 256), dtype=torch.int32, **z)
        self.rt_cnt = torch.zeros(2 * E.NRT, dtype=torch.int32, **z)
        self.work = torch.zeros(4, dtype=torch.int32, **z)
        self.outp = torch.zeros(E.MAXM * E.TOPK_MAX * dim, dtype=torch.bfloat16, **z)
        self.hq = torch.zeros(E.MAXP * nkt2 * E.NMAX * 128, dtype=torch.uint8, **z)
        self.hs = torch.zeros(E.MAXP * E.NMAX, dtype=torch.float32, **z)
        self.h_flags = torch.zeros(E.MAXP, dtype=torch.int32, **z)
        self.pan_cnt = torch.zeros(2 * E.MAXP, dtype=torch.int32, **z)
        self.xg = torch.zeros((E.MAXE + 8 * E.MAXP) * dim, dtype=torch.uint8, **z)
        self.xg_cnt = torch.zeros(2 * E.MAXP, dtype=torch.int32, **z)
        self.fc2_next = torch.zeros(4, dtype=torch.int32, **z)
        self.xqw = torch.zeros(max(E.MAXM, grid * 16) * dim, dtype=torch.uint8, **z)
        self.xsw = torch.zeros(E.MAXM, dtype=torch.float32, **z)
        self.xflag = torch.zeros(E.MAXM, dtype=torch.int32, **z)
        self.seq = 0
        self.gids = torch.zeros(E.MAXM * 8, dtype=torch.int32, **z)
        self.grw = torch.zeros(E.MAXM * 8, dtype=torch.float32, **z)
        self.gws = torch.zeros(E.MAXM * nexp + E.MAXM + 8, dtype=torch.int32, **z)
        self.xqg = torch.zeros(E.MAXM * dim, dtype=torch.uint8, **z)
        self.xsT = torch.zeros(32 * E.MAXM, dtype=torch.float32, **z)
        self.hs4 = torch.zeros(E.MAXP * nkt2 * E.NMAX, dtype=torch.float32, **z)
        self.shout = torch.zeros(E.MAXM * dim, dtype=torch.bfloat16, **z)
        self.rs_ptrs, self.ag_ptrs = [], []
        self.rs_bytes, self.ag_bytes = comm_bytes(E)
        if alloc_comm:
            self.rs_buf = torch.zeros(self.rs_bytes // 4, dtype=torch.int32, **z)
            self.ag_buf = torch.zeros(self.ag_bytes // 4, dtype=torch.int32, **z)

    def link_ptrs(self, rs_ptrs, ag_ptrs):
        self.rs_ptrs = [int(p) for p in rs_ptrs]
        self.ag_ptrs = [int(p) for p in ag_ptrs]


_EMPTY = {}


def _empty(device, dtype):
    key = (str(device), dtype)
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, dtype=dtype, device=device)
    return _EMPTY[key]


def adapt_weights(E, w):
    """tail builds with TP2_W2_TRIM (NSL2 < Humming's slice count): drop the zero K-pad slices of w2 / s2"""
    nsl = int(getattr(E, 'NSL2', 0))
    if nsl and w['w2'].dim() == 3 and w['w2'].shape[1] > nsl:
        assert not w['w2'][:, nsl:].any(), 'w2 pad slices are not zero'
        w = dict(w)
        w['w2'] = w['w2'][:, :nsl].contiguous()
        w['s2'] = w['s2'][:, :nsl].contiguous()
    return w


def run(E, st, w, ids, rw, factor, out, my_rank, ndev, xb, dbg=None, tl=None):
    """routed-only: xb BF16 [M, DIM], ids int32 [M, k], rw fp32 [M, k] -> out BF16 [M, DIM] (all-reduced over ndev; ndev 1: rank-local)"""
    st.seq += 1
    dev = out.device
    eu8, ef32, ebf = _empty(dev, torch.uint8), _empty(dev, torch.float32), _empty(dev, torch.bfloat16)
    args = [st.xqw, st.xsw, w['w1'], w['s1'], w['f1'], w['w2'], w['s2'], w['f2'], ids, rw, ebf, float(factor),
            st.act, st.amax, st.part, st.act_flags, st.part_flags, st.rt_cnt, st.work, st.outp,
            st.hq, st.hs, st.h_flags, st.pan_cnt, st.xg, st.xg_cnt, st.fc2_next,
            st.rs_ptrs if ndev > 1 else [], st.ag_ptrs if ndev > 1 else [], out, my_rank, ndev,
            dbg if dbg is not None else _empty(dev, torch.int64), tl if tl is not None else _empty(dev, torch.int64), xb, st.xflag, st.seq,
            ef32, ef32, ebf, st.gws, eu8, ef32, eu8, ef32, st.xqg, st.xsT, st.hs4, st.shout]
    if hasattr(E, 'NSL2'):   # tail-weight ABI (port_v41 family)
        args += [w.get('w1t', eu8), w.get('s1t', eu8), w.get('f1t', ef32)]
        if hasattr(E, 'T4'):
            args += [w.get('w1t4', eu8), w.get('s1t4', eu8), w.get('f1t4', ef32)]
    E.run(*args)
    return out
