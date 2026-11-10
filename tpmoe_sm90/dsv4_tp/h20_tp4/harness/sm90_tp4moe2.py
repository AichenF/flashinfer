"""Shared adapter body for the SM90 tp4moe2-family TP kernels (h123 TP8, tp4moe2 v26 TP4, v94ex TP4): the package's own host module
(tp4moe2.py: RankState / run), extensions either prebuilt (.so, M-dispatched) or built from the package source with the package's
own flags, weights in Humming's layout prepared from the canonical MXFP4 TP slices (humming_prep), RS / AG all-reduce buffers by
cudaMalloc + CUDA IPC as the serving hook does.  Routed-only: run(..., ids, rw, shared=None, factor=1.0, xb=bf16 x) -> the kernel
quantizes x (Humming per-token e4m3), runs the routed experts and the in-kernel LL all-reduce."""
import hashlib
import importlib.machinery
import importlib.util
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_so(path):
    path = str(path)
    name = Path(path).stem
    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def build_ext(src_dir, src, name, flags, build_root):
    from torch.utils.cpp_extension import load
    bdir = Path(build_root) / name
    bdir.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    return load(name=name, sources=[str(Path(src_dir) / src)], build_directory=str(bdir),
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-I{src_dir}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK', *flags],
                verbose=False)


class Tp4moe2Cell:
    """exts: [(lo, hi, module)] (first = default); host: the package's tp4moe2.py module"""
    xq_mode = 'tok'

    def __init__(self, ctx, name, host_path, exts, provenance):
        self.ctx, self.name = ctx, name
        H, I, E, K = ctx.shape
        assert (H, E, K) == (4096, 256, 6), 'tp4moe2 family: DSV4-Flash only'
        self.T = load_module(host_path, f'tp4moe2_host_{abs(hash(str(host_path)))}')
        self.exts = exts
        E0 = exts[0][2]
        assert int(getattr(E0, 'NDEV', 4)) == ctx.ndev, (int(getattr(E0, 'NDEV', 4)), ctx.ndev)
        self.T._ext = E0
        self.max_m = int(E0.MAXM)
        self.prov = provenance
        self.st = self.T.RankState(ctx.dev, alloc_comm=False)
        ptrs = [ctx.ipc_alloc(nb) for nb in (self.T.RankState.RS_BYTES(), self.T.RankState.AG_BYTES())]
        self.st.link_ptrs(*ptrs)
        t0 = time.time()
        from humming_prep import humming_tp_weights
        self.w = humming_tp_weights(ctx.tp_w(), H, I // ctx.ndev, ctx.dev)
        ctx.log(f'  {name}: Humming weights {time.time() - t0:.0f} s, MAXM {self.max_m}')

    def ext_of(self, m):
        for lo, hi, e in self.exts[1:]:
            if lo <= m <= hi:
                return e
        return self.exts[0][2]

    def bind(self, M, x, ids, w, out):
        T, st, wts, rank, N = self.T, self.st, self.w, self.ctx.rank, self.ctx.ndev
        e = self.ext_of(M)

        def fn():
            saved = T._ext
            T._ext = e
            try:
                T.run(st, wts, None, None, ids, w, None, 1.0, out, rank, N, xb=x)
            finally:
                T._ext = saved
        return fn

    def pre_replay(self):   # the host launch tag (seq) is frozen inside a one-call graph: clear the per-token quant flags
        self.st.xflag.zero_()

    def info(self):
        return self.prov


def so_info(paths):
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def build_shared(ctx, src_dir, src, name, flags):
    """rank 0 builds (node-local build root: $SUITE_BUILD), then every rank loads the same build"""
    root = os.environ.get('SUITE_BUILD', '/tmp/tpsuite_build')
    if ctx.rank == 0:
        build_ext(src_dir, src, name, flags, root)
    dist.barrier(group=ctx.gloo)
    return build_ext(src_dir, src, name, flags, root)
