"""Host side of tpmoe_b2 (B200 TP4 fused MoE): extension build, per-rank workspace + LL all-reduce buffers, launch wrapper."""
import os
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
_EXT = {}
FC2_ROW_OF_LANE = (0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31,
                   32, 36, 40, 44, 48, 52, 56, 60, 33, 37, 41, 45, 49, 53, 57, 61, 34, 38, 42, 46, 50, 54, 58, 62, 35, 39, 43, 47, 51, 55, 59, 63,
                   64, 68, 72, 76, 80, 84, 88, 92, 65, 69, 73, 77, 81, 85, 89, 93, 66, 70, 74, 78, 82, 86, 90, 94, 67, 71, 75, 79, 83, 87, 91, 95,
                   96, 100, 104, 108, 112, 116, 120, 124, 97, 101, 105, 109, 113, 117, 121, 125, 98, 102, 106, 110, 114, 118, 122, 126, 99, 103,
                   107, 111, 115, 119, 123, 127)   # SGLang trtllm-gen w2 layout: output row (inside a 128-row tile) of TMEM lane (b2_umma_test)


def load_ext(src='tpmoe_b2.cu', tag='', flags=()):
    key = (src, tag, tuple(flags))
    if key in _EXT:
        return _EXT[key]
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    build = OUT / f'build_tpb2{tag}'
    build.mkdir(exist_ok=True)
    m = load(name=f'tpmoe_b2{tag}', sources=[str(OUT / src)], build_directory=str(build),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK', *flags],
             extra_ldflags=['-lcuda'], verbose=False)
    _EXT[key] = m
    return m


class RankState:
    """Per-rank workspace of the kernel (+ the IPC-linked LL buffers when ndev > 1)."""

    def __init__(self, ext, dev, ndev=4, rank=0):
        self.E, self.dev, self.ndev, self.rank = ext, dev, ndev, rank
        E = ext
        z = dict(device=dev)
        self.xq = torch.zeros(E.MAXM * 4096, dtype=torch.uint8, **z)
        self.xs = torch.zeros(E.MAXM * 128, dtype=torch.uint8, **z)
        self.xg = torch.zeros(E.MAXP * E.NCH1 * E.NMAX * 128, dtype=torch.uint8, **z)
        self.xgsf = torch.zeros(E.MAXP * E.NSTG1 * 512, dtype=torch.uint8, **z)
        self.dbg = None
        self.himg = torch.zeros(E.MAXP * E.NCH2 * E.NMAX * 128, dtype=torch.uint8, **z)
        self.hsf = torch.zeros(E.MAXP * E.NSTG2 * 512, dtype=torch.uint8, **z)
        self.part = torch.zeros(E.MAXG * 2 * E.NMAX * 128, dtype=torch.float32, **z)
        self.xgs = torch.zeros(E.MAXPS * E.NCH1 * E.NMAX * 128, dtype=torch.uint8, **z)   # fused shared expert workspace
        self.xgsfs = torch.zeros(E.MAXPS * E.NSTG1 * 512, dtype=torch.uint8, **z)
        self.hsimg = torch.zeros(E.MAXPS * E.NCH2 * E.NMAX * 128, dtype=torch.uint8, **z)
        self.hssf = torch.zeros(E.MAXPS * E.NSTG2 * 512, dtype=torch.uint8, **z)
        self.shout = torch.zeros(E.MAXM * 4096, dtype=torch.bfloat16, **z)
        self.part_flags = torch.zeros(E.MAXG, dtype=torch.int32, **z)
        self.xflag = torch.zeros(2 * E.MAXM, dtype=torch.int32, **z)
        self.gcnt = torch.zeros(2, dtype=torch.int32, **z)
        self.pan_cnt = torch.zeros(2 * E.MAXP, dtype=torch.int32, **z)
        self.rdy = torch.zeros(2 * (E.MAXP + 32), dtype=torch.int32, **z)
        self.fc2_next = torch.zeros(4, dtype=torch.int32, **z)
        self.rt_cnt = torch.zeros(2 * E.NRT, dtype=torch.int32, **z)
        self.work = torch.zeros(4, dtype=torch.int32, **z)
        self.outp = torch.zeros(E.MAXM * 8 * 4096, dtype=torch.bfloat16, **z)
        self.ids = torch.zeros(E.MAXM * 6, dtype=torch.int32, **z)
        self.rw = torch.zeros(E.MAXM * 6, dtype=torch.float32, **z)
        self.lanes = torch.tensor(FC2_ROW_OF_LANE, dtype=torch.uint8, device=dev)
        self.rs, self.ag = [], []
        self.grid = torch.cuda.get_device_properties(dev).multi_processor_count

    @staticmethod
    def RS_BYTES(E):
        return 2 * int(E.RS_PAR) * 16   # two epoch-parity halves of uint4 messages

    @staticmethod
    def AG_BYTES(E):
        return 2 * int(E.AG_PAR) * 16

    def link_ptrs(self, rs, ag):
        self.rs, self.ag = list(rs), list(ag)


class Weights:
    """The layer's SGLang Mxfp4FlashinferTrtllmMoEMethod tensors (in place) + the cached TMA maps."""

    def __init__(self, ext, w13, w13_sf, w2, w2_sf):
        u8 = lambda t: t.view(torch.uint8)
        self.w13, self.w13_sf, self.w2, self.w2_sf = u8(w13), u8(w13_sf).contiguous(), u8(w2), u8(w2_sf).contiguous()
        self.maps = ext.encode_maps(self.w13, self.w2)


class SharedWeights:
    """The layer's shared-expert slice as SGLang's Fp8LinearMethod holds it on SM100 (DeepGEMM): e4m3 gate_up [1024, 4096] / down
    [4096, 512] used in place, + the SFA images of their UE8M0 128 x 128 block scales (gs [8, 32] / ds [32, 4] float powers of two)."""

    def __init__(self, ext, gu_w, gs, dn_w, ds):
        self.gu_w, self.dn_w = gu_w.view(torch.uint8), dn_w.view(torch.uint8)
        self.maps = ext.encode_maps_shared(self.gu_w, self.dn_w)

        def images(sc):   # [row blocks, k blocks] powers of two -> [rb][kb][512] UE8M0 bytes (every row and k32 block of the 128 x 128 block)
            e = (torch.log2(sc.float()).round() + 127).to(torch.uint8)
            assert torch.equal(torch.exp2(e.float() - 127), sc.float()), 'shared scales must be powers of two (UE8M0)'
            return e.unsqueeze(-1).expand(*e.shape, 512).contiguous()
        self.sfa1 = images(gs).to(gu_w.device)
        self.sfa2 = images(ds).to(gu_w.device)


def run(st, W, x, out, glog=None, gbias=None, ids=None, rw=None, shared=None, factor=1.5, SW=None):
    """out [M, 4096] bf16 = all-reduce over the TP group of (shared + factor * routed); fused gate when glog is given"""
    E = st.E
    M = x.shape[0]
    if glog is not None:
        ids_t, rw_t = st.ids[:M * 6].view(M, 6), st.rw[:M * 6].view(M, 6)
    else:
        ids_t, rw_t = ids, rw
    E.run(W.maps, W.w13_sf, W.w2_sf, st.lanes, x, glog, gbias, ids_t, rw_t, shared, float(factor), st.xq, st.xs, st.xg, st.xgsf, st.dbg, st.himg, st.hsf,
          st.part, st.part_flags, st.xflag, st.gcnt, st.pan_cnt, st.rdy, st.fc2_next, st.rt_cnt, st.work, st.outp, st.rs, st.ag, out,
          st.rank, st.ndev, st.grid, SW.maps if SW else None, SW.sfa1 if SW else None, SW.sfa2 if SW else None,
          st.xgs, st.xgsfs, st.hsimg, st.hssf, st.shout)
    return out
