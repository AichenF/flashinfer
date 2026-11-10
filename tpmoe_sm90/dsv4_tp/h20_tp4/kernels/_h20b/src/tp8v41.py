"""Host side of tp8v41.cu (port_v41.py): extension loader (prebuilt .so or a build), per-rank workspace (+ LL all-reduce buffers),
run() for the routed-only path (ids / weights given, BF16 x quantized in the kernel, in-kernel all-reduce).  Sizes come from the
extension's exported constants (DIM, INTER, NQG, NKT2, RS_BYTES, ...): one wrapper for the DSV4-Flash and the DSV4.1 builds."""
import importlib.machinery
import importlib.util
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def build(src='tp8v41.cu', model=1, ndev=8, shared=0, extra=(), tag=None, verbose=False):
    """build (or load the cached build of) src with -DTP2_MODEL / -DTP2_NDEV / -DTP2_SHARED; returns the module"""
    from torch.utils.cpp_extension import load
    tag = tag or f'm{model}_n{ndev}_s{shared}' + ''.join('_' + ''.join(c if c.isalnum() else '_' for c in f) for f in extra)
    stem = Path(src).stem.replace('.', '_')
    name = f'{stem}_{tag}'
    bdir = Path(os.environ.get('TP_BUILD_ROOT', str(HERE))) / f'build_{name}'   # (h20b: node-local build root)
    bdir.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    return load(name=name, sources=[str(HERE / src)], build_directory=str(bdir),
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-I{HERE}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
                                   f'-DTP2_MODEL={model}', f'-DTP2_NDEV={ndev}', f'-DTP2_SHARED={shared}', *extra], verbose=verbose)


def load_so(path):
    """a prebuilt extension .so (module name = file stem)"""
    path = str(path)
    name = Path(path).stem
    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class RankState:
    """Per-rank workspace for extension E.  Counters / flags are epoch-tagged or parity-reset by the kernel itself."""

    def __init__(self, E, device, alloc_comm=True):
        self.E = E
        dim = getattr(E, 'DIM', 4096)
        inter, nqg, nkt2 = E.INTER, E.NQG, E.NKT2
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
        self.gws = torch.zeros(E.MAXM * getattr(E, 'NEXP', 256) + E.MAXM + 8, dtype=torch.int32, **z)
        self.xqg = torch.zeros(E.MAXM * dim, dtype=torch.uint8, **z)
        self.xsT = torch.zeros(32 * E.MAXM, dtype=torch.float32, **z)
        self.hs4 = torch.zeros(E.MAXP * nkt2 * E.NMAX, dtype=torch.float32, **z)
        self.shout = torch.zeros(E.MAXM * dim, dtype=torch.bfloat16, **z)
        self.rs_ptrs, self.ag_ptrs = [], []
        self.rs_bytes = int(getattr(E, 'RS_BYTES', E.NRT * E.MAXM * E.NMSG * 16))
        self.ag_bytes = int(getattr(E, 'AG_BYTES', E.NRT * E.MAXM * E.NMSG * 16))
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
    """v41 builds with TP2_W2_TRIM (NSL2 9): drop Humming's K pad slice 9 of w2 / s2 (zero codes: checked)"""
    nsl = int(getattr(E, 'NSL2', 0))
    if nsl and w['w2'].dim() == 3 and w['w2'].shape[1] > nsl:
        assert not w['w2'][:, nsl:].any(), 'w2 pad slices are not zero'
        w = dict(w)
        w['w2'] = w['w2'][:, :nsl].contiguous()
        w['s2'] = w['s2'][:, :nsl].contiguous()
    return w


def run(E, st, w, ids, rw, factor, out, my_rank, ndev, xb, tail=None, old_abi=False, dbg=None, tl=None):
    """routed-only: xb BF16 [M, DIM], ids int32 [M, k], rw fp32 [M, k]; w = dict(w1, s1, f1, w2, s2, f2 [, w1t, s1t, f1t]).
    old_abi: an h123 .so (no tail arguments).  Returns out (BF16 [M, DIM]: rank-local routed x factor if ndev 1, else all-reduced)."""
    st.seq += 1
    dev = out.device
    eu8, ef32, ebf = _empty(dev, torch.uint8), _empty(dev, torch.float32), _empty(dev, torch.bfloat16)
    args = [st.xqw, st.xsw, w['w1'], w['s1'], w['f1'], w['w2'], w['s2'], w['f2'], ids, rw, ebf, float(factor),
            st.act, st.amax, st.part, st.act_flags, st.part_flags, st.rt_cnt, st.work, st.outp,
            st.hq, st.hs, st.h_flags, st.pan_cnt, st.xg, st.xg_cnt, st.fc2_next,
            st.rs_ptrs if ndev > 1 else [], st.ag_ptrs if ndev > 1 else [], out, my_rank, ndev,
            dbg if dbg is not None else _empty(dev, torch.int64), tl if tl is not None else _empty(dev, torch.int64), xb, st.xflag, st.seq,
            ef32, ef32, ebf, st.gws, eu8, ef32, eu8, ef32, st.xqg, st.xsT, st.hs4, st.shout]
    if not old_abi:
        args += [w.get('w1t', eu8), w.get('s1t', eu8), w.get('f1t', ef32)]
        if hasattr(E, 'T4'):   # (phase-2 ABI) the T4 split-tail weights
            args += [w.get('w1t4', eu8), w.get('s1t4', eu8), w.get('f1t4', ef32)]
    E.run(*args)
    return out
