"""Host side of tp8v41 (DeepSeek-V4.1-Flash routed MoE, TP8, B300): build, per-rank weight slices in the kernel's layouts, workspace,
launch wrapper, torch references (the kernel's numerics; bf16 dense like gc-moe's reference_moe_forward), routings."""
import os
import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_EXT = {}
DIM, INTER_FULL, NEXP, TOPK, NDEV = 5120, 2304, 384, 6, 8
INTER = INTER_FULL // NDEV   # 288 per rank
NT1, NRT2, NTL = 5, DIM // 128, DIM // 512
CLAMP = 10.0


def load_ext(src='tp8v41.cu', tag='', flags=(), verbose=False):
    src = os.environ.get('TP8_SRC', src)   # (override: another tp8v41-lineage source in this dir, e.g. an imported variant)
    tag = re.sub(r'[^A-Za-z0-9_]', '_', tag) + ('' if src == 'tp8v41.cu' else '_' + re.sub(r'[^A-Za-z0-9]', '', src)[:16])
    flags = tuple(flags) + tuple(os.environ.get('TP8_FLAGS', '').split())
    key = (src, tag, flags)
    if key in _EXT:
        return _EXT[key]
    import hashlib
    tag = f'{tag}_{hashlib.md5((" ".join(flags) + src).encode()).hexdigest()[:8]}'
    from torch.utils.cpp_extension import load
    maj, mnr = torch.cuda.get_device_capability()
    arch = f'{maj}{mnr}a'
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{maj}.{mnr}a'
    build = Path(os.environ.get('TP8_BUILD', str(HERE / 'build'))) / f'b_{tag}'
    build.mkdir(parents=True, exist_ok=True)
    name = f'tp8v41{tag}'
    m = load(name=name, sources=[str(HERE / src)], build_directory=str(build),
             extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
                                f'-DTPB2_NS=ns_{tag}', *flags],
             extra_include_paths=[str(HERE)], extra_ldflags=['-lcuda'], verbose=verbose)
    _EXT[key] = m
    return m


SPEC_COMMON = ('-DTPB2_SPEC=1', '-DTPB2_WHOLE=1', '-DTPB2_RS=0', '-DTPB2_QX1=1', '-DTPB2_PUT1=1', '-DTPB2_CEPOCH=1',
               '-DTPB2_NMAX=24', '-DTPB2_NS_STAGES=5', '-DTPB2_EARLYW=2', '-DTPB2_IDV=1', '-DTPB2_XRLX=1', '-DTPB2_EPI2=1',
               '-DTPB2_LLS=1')   # (LLS kept 11-09: o35 dec 2 reps M256/384/512 -5.0/-6.3/-9.5 and -5.7/-7.0/-9.5 us; the RS buffer's LL region starts 0xFF)


NMAX32_MINM = 256   # (per-M default: builds with MMIN >= this use NMAX 32)
F2H_MINM = int(os.environ.get('TP8_F2H_MINM', 128))   # ((v41 H2) per-M: builds with MMIN >= this shard FC2 on H (FC1 h exchange, local T1, all-gather)); 128
                                                     # with the T1PAR tail: o62 M64 +2.4/+2.1, M128 -5.4/-5.8 us vs base; o61 T1PAR -5.1..-5.9 at M256-512
WS_NMAX = 32        # (workspace sized for the largest panel width of any build)


def spec_flags(mlo, mhi, ndev, grid=148, extra=()):
    """compile-time specialized build for mlo <= M <= mhi at TP size ndev; an extra -DX=.. replaces a default -DX (nvcc keeps the first)"""
    ex = tuple(extra)
    over = {f.split('=')[0] for f in ex if f.startswith('-D')}
    base = (f'-DTPB2_MMIN={mlo}', f'-DTPB2_MMAX={mhi}', f'-DTPB2_NDEV={ndev}', f'-DTPB2_GRID={grid}') + SPEC_COMMON
    if mlo >= NMAX32_MINM:   # (opt round 1: 32-token panels for M >= 256 -- fewer re-streamed hot-expert panels; dec M512 -4..-5 us, M256 -1)
        base = tuple('-DTPB2_NMAX=32' if f == '-DTPB2_NMAX=24' else f for f in base)
    if mlo >= F2H_MINM and ndev == 8:   # ((v41 H2) FC2 sharded on H at large M)
        base = base + ('-DTPB2_F2H=1',)
    return tuple(f for f in base if f.split('=')[0] not in over) + ex


# ------------------------------------------------------------------------------------------------ weights
def synth_slices(rank, dev, seed=7, scale=0.05):
    """this rank's bf16 slices: w1 [E][576][DIM] = [gate rows 288 r ..; up rows 288 r ..] of w13 = [gate; up] (gc-moe order), w2 [E][DIM][288]"""
    g = torch.Generator(device=dev).manual_seed(seed + 1000 * rank)
    w1 = torch.randn(NEXP, 2 * INTER, DIM, dtype=torch.bfloat16, device=dev, generator=g) * scale
    w2 = torch.randn(NEXP, DIM, INTER, dtype=torch.bfloat16, device=dev, generator=g) * scale
    return w1, w2


_E2M1 = None


def _e2m1_table(dev):
    global _E2M1
    if _E2M1 is None or _E2M1.device != dev:
        _E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=dev)
    return _E2M1


def quant_fp4(x):
    """MXFP4 (DeepGEMM per_token_cast_to_fp4(use_ue8m0=True, gran_k=32) when available): rows of x [R, K] -> packed uint8 [R, K/2] (low nibble
    = even element), E8M0 bytes [R, K/32]"""
    R, K = x.shape
    try:
        from deep_gemm.utils import per_token_cast_to_fp4
        packed, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
        e8 = (torch.log2(sf.float()).round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
        return packed.view(torch.uint8).contiguous(), e8.contiguous()
    except ImportError:
        pass
    xv = x.float().view(R, K // 32, 32)
    am = xv.abs().amax(-1).clamp_min(1e-4)
    ex = torch.ceil(torch.log2(am / 6.0)).to(torch.int32)
    s = torch.exp2(ex.float()).unsqueeze(-1)
    tab = _e2m1_table(x.device)[:8]
    a = (xv / s).abs().clamp(max=6.0)
    code = (a.unsqueeze(-1) - tab).abs().argmin(-1).to(torch.uint8) | ((xv < 0).to(torch.uint8) << 3)
    code = code.view(R, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, (ex + 127).to(torch.uint8).contiguous()


def dq_fp4(packed, e8):
    """packed e2m1 [R, K/2] x E8M0 [R, K/32] -> fp32 [R, K]"""
    R = packed.shape[0]
    v = torch.stack([packed & 15, packed >> 4], -1).reshape(R, -1).long()
    return _e2m1_table(packed.device)[v] * torch.exp2(e8.float() - 127.0).repeat_interleave(32, dim=1)


def quantize_slices(w1, w2, chunk=32):
    """bf16 slices -> MXFP4 raw (source row order): w13 [E][576][2560] + s13 [E][576][160]; w2 [E][DIM][144] + s2 [E][DIM][9]"""
    E = w1.shape[0]
    q1, s1, q2, s2 = [], [], [], []
    for e0 in range(0, E, chunk):
        a, b = quant_fp4(w1[e0:e0 + chunk].reshape(-1, DIM))
        q1.append(a.view(-1, 2 * INTER, DIM // 2)); s1.append(b.view(-1, 2 * INTER, DIM // 32))
        a, b = quant_fp4(w2[e0:e0 + chunk].reshape(-1, INTER))
        q2.append(a.view(-1, DIM, INTER // 2)); s2.append(b.view(-1, DIM, INTER // 32))
    return dict(w13=torch.cat(q1), s13=torch.cat(s1), w2=torch.cat(q2), s2=torch.cat(s2))


def w13_perm(dev):
    """stored row s of the kernel's FC1 tile layout <- source row (gate 0..287 | up 288..575): 32-row blocks of interleaved (up_j, gate_j) pairs,
    lane l of a block = interleaved row (l % 8) 4 + l / 8 (the trtllm-gen epilogue-tile shuffle the epilogue's lane pairing expects)"""
    s = torch.arange(2 * INTER)
    lane = s % 32
    ii = 32 * (s // 32) + (lane % 8) * 4 + lane // 8
    pair, up = ii // 2, (ii % 2) == 0
    return torch.where(up, INTER + pair, pair).to(dev)


def _sf_images(sf_rows, n_tiles, n_chunks):
    """E8M0 [E][rows][4 n_chunks] (rows <= 128 n_tiles, zero padded) -> block_scale_interleave images [E][n_tiles][n_chunks][512]"""
    E, R, KB = sf_rows.shape
    pad = torch.zeros(E, 128 * n_tiles, KB, dtype=torch.uint8, device=sf_rows.device)
    pad[:, :R] = sf_rows
    v = pad.view(E, n_tiles, 4, 32, n_chunks, 4)          # [e][t][g = r / 32][i = r % 32][c][kb]
    return v.permute(0, 1, 4, 3, 2, 5).contiguous()        # [e][t][c][i][g][kb]: byte i 16 + g 4 + kb


def kernel_layout(raw):
    """raw MXFP4 slices -> the kernel's tensors: w13 [E][576][2560] (+ SF images [E][5][40][512]), w2 main [E][DIM][128] (+ [E][40][2][512]),
    w2 tail [E][128][640] (+ [E][10][512])"""
    E, dev = raw['w13'].shape[0], raw['w13'].device
    perm = w13_perm(dev)
    w13 = raw['w13'][:, perm].contiguous()
    s13 = _sf_images(raw['s13'][:, perm], NT1, DIM // 128)
    w2m = raw['w2'][:, :, :128].contiguous()
    w2t = raw['w2'][:, :, 128:144].reshape(E, NRT2, 128, 16).permute(0, 2, 1, 3).reshape(E, 128, NRT2 * 16).contiguous()
    s2m = _sf_images(raw['s2'][:, :, :8], NRT2, 2)
    st = raw['s2'][:, :, 8].reshape(E, NTL, 4, 4, 32)      # [e][c4][kq][g][i]
    s2t = st.permute(0, 1, 4, 3, 2).contiguous()            # [e][c4][i][g][kq]
    # (F1F) folded tile 4: rows r < 64 = stored row 512 + r over K 0..2559, rows 64 + r = the same row over K 2560..5119 (+ SF images [E][20][512])
    t4, s4 = w13[:, 512:576], raw['s13'][:, perm][:, 512:576]
    w13f = torch.cat([t4[:, :, :DIM // 4], t4[:, :, DIM // 4:]], 1).contiguous()
    s13f = _sf_images(torch.cat([s4[:, :, :DIM // 64], s4[:, :, DIM // 64:]], 1).contiguous(), 1, DIM // 256)
    return dict(w13=w13, s13=s13.view(E, -1), w2m=w2m, w2t=w2t, s2m=s2m.view(E, -1), s2t=s2t.view(E, -1), w13f=w13f, s13f=s13f.view(E, -1),
                lanes=torch.arange(128, dtype=torch.uint8, device=dev))


def stage_blocks(w, rows_pad=None):
    """(W13C) [E][R][K/2] rows of a 128-row tile (R <= 128) -> [E][K/256 stages][R][128 B]: every 2-chunk stage contiguous"""
    E, R, KB = w.shape
    return w.view(E, R, KB // 128, 128).permute(0, 2, 1, 3).contiguous()


class Weights:
    def __init__(self, ext, kl):
        self.kl = kl
        self.w13, self.w2m, self.w2t = kl['w13'], kl['w2m'], kl['w2t']
        self.w13_sf, self.w2_sf, self.w2t_sf, self.lanes = kl['s13'], kl['s2m'], kl['s2t'], kl['lanes']
        self.w13f, self.w13f_sf = kl['w13f'], kl['s13f']
        self.w13h = None
        if int(getattr(ext, 'W13C', 0)):   # (stage-contiguous FC1 blocks: tiles 0..3, tile 4, folded tile 4)
            E = self.w13.shape[0]
            self.w13 = torch.stack([stage_blocks(self.w13[:, 128 * t:128 * (t + 1)]) for t in range(4)], 1).contiguous()
            self.w13h = stage_blocks(kl['w13'][:, 512:576])
            self.w13f = stage_blocks(kl['w13f'])
        self.maps = ext.encode_maps(self.w13, self.w2m, self.w2t, self.w13f, self.w13h)
        if int(getattr(ext, 'F2H', 0)):   # ((v41 H2) F2H) this rank's H-sharded FC2 weights: kl['f2h'] = f2h_weights(..) (TP8), else synthetic (timing)
            dev = self.w13.device
            if 'f2h' in kl:
                self.f2h_w2, self.f2h_s2 = kl['f2h']
            else:
                g = torch.Generator(device=dev).manual_seed(4242)
                self.f2h_w2 = torch.randint(0, 256, (NEXP, DIM // NDEV, INTER_FULL // 2), dtype=torch.uint8, device=dev, generator=g)
                self.f2h_s2 = torch.full((NEXP, DIM // NDEV, INTER_FULL // 32), 118, dtype=torch.uint8, device=dev)
            self.w2h = self.f2h_w2
            self.w2h_sf = _sf_images(self.f2h_s2, (DIM // NDEV) // 128, INTER_FULL // 128).reshape(-1).contiguous()


# ------------------------------------------------------------------------------------------------ workspace
class RankState:
    def __init__(self, ext, dev, ndev=1, rank=0):
        E = self.E = ext
        self.dev, self.ndev, self.rank = dev, ndev, rank
        z = dict(device=dev)
        u8 = dict(dtype=torch.uint8, **z)
        i32 = dict(dtype=torch.int32, **z)
        self.xq = torch.zeros(E.MAXM * E.DIM, **u8)
        self.xs = torch.zeros(E.MAXM * E.DIM // 32, **u8)
        ng = True   # (per-panel x / SFB images for GATHER=0 builds; allocated always: one workspace serves every build)
        self.xg = torch.zeros(E.MAXP * E.NCH1 * max(E.NMAX, WS_NMAX) * 128 if ng else 1024, **u8)
        self.xgsf = torch.zeros(E.MAXP * E.NSTG1 * 512 if ng else 1024, **u8)
        self.himg = torch.zeros(E.MAXP * E.NCH2 * max(E.NMAX, WS_NMAX) * 128 + 4096, **u8)
        self.hsf = torch.zeros(E.MAXP * 4 * 512, **u8)
        self.part = torch.zeros(E.MAXG * 4 * max(E.NMAX, WS_NMAX) * 128, dtype=torch.float32, **z)
        self.part_flags = torch.zeros(2 * E.MAXG, **i32)
        self.xflag = torch.zeros(2 * E.MAXM, **i32)
        self.gcnt = torch.zeros(2, **i32)
        self.pan_cnt = torch.zeros(2 * E.MAXP, **i32)
        self.rdy = torch.zeros(2 * (E.MAXP + 32), **i32)
        self.fc2_next = torch.zeros(4, **i32)
        self.rt_cnt = torch.zeros(2 * E.NRT, **i32)
        self.work = torch.zeros(64 + 256, **i32)
        self.outp = torch.zeros(E.MAXM * E.TOPK_MAX * E.OUTP_LD, dtype=torch.bfloat16, **z)
        self.small = torch.zeros(4096, **u8)
        self.shout = torch.zeros(16, dtype=torch.bfloat16, **z)
        self.xsf2 = torch.zeros(E.MAXM * E.NSTG1 * 16, **u8)
        self.tcnt = torch.zeros(2 * E.MAXM * E.NRT + int(E.LL_INTS) + 1024, **i32)
        self.rs = []
        self.grid = int(os.environ.get('TP8_GRID', 0)) or torch.cuda.get_device_properties(dev).multi_processor_count

    @staticmethod
    def RS_BYTES(E):   # (+ TB_BYTES: the suite_b200 tail stage zeroes its TAILB counters past the LL halves in EVERY build, TAILB on or off)
        return 2 * int(E.RS_PAR) * 16 + (1 << 20) + int(getattr(E, 'TB_BYTES', 0))

    def link_ptrs(self, ptrs):
        self.rs = [int(p) for p in ptrs]


def run(st, W, x, ids, rw, out, dbg=None, E=None):
    """out [M, DIM] bf16 = this rank's routed sum (ndev 1) or the TP8 all-reduce of it (ndev 8).  ids int32 [M][6], rw fp32 [M][6]."""
    E = E or st.E
    if int(getattr(E, 'F2H', 0)):   # ((v41 H2) F2H) gathered-h buffers: st.hgp (TP8: symmetric, set by the harness), else one local buffer
        if getattr(st, 'hgp', None) is None:
            st.hg = torch.zeros(int(E.HG_BYTES), dtype=torch.uint8, device=st.dev)   # (zeroed: the epoch ready tags start below every epoch; one layout for every build)
            st.hgp = [st.hg.data_ptr()]
        key = (id(W), id(st), tuple(st.hgp))
        if getattr(E, '_f2h_key', None) != key:   # (the build's tensor maps: encoded once per (weights, state))
            E.set_f2h(W.w2h, W.w2h_sf, st.hgp, st.rank)
            E._f2h_key = key
    rs = st.rs if st.ndev > 1 else []
    E.run(W.maps, W.w13_sf, W.w2_sf, W.lanes, x, None, None, ids, rw, None, 1.0, st.xq, st.xs, st.xg, st.xgsf, dbg, st.himg, st.hsf,
          st.part, st.part_flags, st.xflag, st.gcnt, st.pan_cnt, st.rdy, st.fc2_next, st.rt_cnt, st.work, st.outp, rs, rs, out,
          st.rank, st.ndev, st.grid, None, W.w13f_sf, W.w2t_sf, st.small, st.small, st.small, st.small, st.shout, st.xsf2, st.tcnt,
          [W.w13.data_ptr(), W.w2m.data_ptr(), 0, 0], None, None, None, 1e-5)
    return out


# ------------------------------------------------------------------------------------------------ references
def e4m3(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


def _fbits(x):
    return x.contiguous().view(torch.int32)


def mxfp8_x(x):
    """x fp32 [M, K] -> dequantized MXFP8, scale 2^ceil(log2(amax * fp32(1/448))) (smallest normal for 0) = the kernel's quant_x"""
    M, K = x.shape
    xb = x.view(M, K // 32, 32)
    am = xb.abs().amax(-1) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    am = torch.maximum(am, torch.tensor(1.17549435e-38, device=x.device))
    b = _fbits(am)
    ex = ((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0).int()
    s = torch.exp2(ex.float()).unsqueeze(-1)
    return (e4m3(xb / s) * s).view(M, K)


def mxfp8_h(a):
    """FC1 activation -> dequantized MXFP8 with the OCP rule 2^(floor(log2 amax) - 8) per 32 (the kernel's ue8m0_exp_floor)"""
    M, K = a.shape
    ab = a.view(M, K // 32, 32)
    am = ab.abs().amax(-1)
    ex = torch.clamp(((_fbits(am) >> 23) & 0xFF) - 127 - 8, min=-127)
    s = torch.exp2(ex.float()).unsqueeze(-1)
    return (e4m3(ab * torch.exp2(-ex.float()).unsqueeze(-1)) * s).view(M, K)


def swiglu(g, u):
    g = g.clamp(max=CLAMP)
    u = u.clamp(-CLAMP, CLAMP)
    return u * (g * torch.sigmoid(g))


def bf16r(x):
    return x.to(torch.bfloat16).float()


def ref_rank(raw, x, ids, rw):
    """one rank, the kernel's numerics: bf16(sum_k w_k bf16(FC2(MXFP8(SwiGLU(FC1(MXFP8(x))))))) with fp32 weights, slots in order"""
    M = x.shape[0]
    xq = mxfp8_x(x.float())
    y = torch.zeros(M, ids.shape[1], DIM, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not (0 <= e < NEXP):
            continue
        tk = (ids == e).nonzero()
        w1 = dq_fp4(raw['w13'][e], raw['s13'][e])
        w2 = dq_fp4(raw['w2'][e], raw['s2'][e])
        gu = xq[tk[:, 0]] @ w1.t()
        h = mxfp8_h(swiglu(gu[:, :INTER], gu[:, INTER:]))
        y[tk[:, 0], tk[:, 1]] = bf16r(h @ w2.t())
    acc = torch.zeros(M, DIM, dtype=torch.float32, device=x.device)
    for k in range(ids.shape[1]):
        acc = acc + y[:, k] * rw[:, k].float().unsqueeze(1)
    return bf16r(acc)


def f2h_weights(raw, rank):
    """((v41 H2) F2H) this rank's H-sharded FC2 weights from every rank's canonical MXFP4 TP slices (all-gathered): rows 640 rank .. of
    the full quantized W2 (K slices concatenated: per-32 scales, so the full-K quantization IS the slices') -> fp4 [E][640][1152], E8M0 [E][640][72]"""
    import torch.distributed as dist
    w2s = [torch.empty_like(raw['w2']) for _ in range(NDEV)]
    dist.all_gather(w2s, raw['w2'].contiguous())
    s2s = [torch.empty_like(raw['s2']) for _ in range(NDEV)]
    dist.all_gather(s2s, raw['s2'].contiguous())
    rows = slice((DIM // NDEV) * rank, (DIM // NDEV) * (rank + 1))
    w2h = torch.cat([w[:, rows] for w in w2s], dim=2).contiguous()
    s2h = torch.cat([v[:, rows] for v in s2s], dim=2).contiguous()
    del w2s, s2s
    return w2h, s2h


def ref_rank_h(raw, w2h, s2h, x, ids, rw, rank):
    """((v41 H2) F2H) one rank's partial with the H2 numerics: h slices (as ref_rank) all-gathered -> full K 2304, FC2 of this rank's 640 rows
    (fp32, one bf16 rounding per slot), slots in order (fp32) -> bf16 in rows 640 rank .., 0 elsewhere (the all-reduce adds zeros)"""
    import torch.distributed as dist
    M, K6 = ids.shape
    xq = mxfp8_x(x.float())
    hl = torch.zeros(M, K6, INTER, dtype=torch.float32, device=x.device)
    ue = [e for e in torch.unique(ids).tolist() if 0 <= e < NEXP]
    for e in ue:
        tk = (ids == e).nonzero()
        w1 = dq_fp4(raw['w13'][e], raw['s13'][e])
        gu = xq[tk[:, 0]] @ w1.t()
        hl[tk[:, 0], tk[:, 1]] = mxfp8_h(swiglu(gu[:, :INTER], gu[:, INTER:]))
    hs = [torch.empty_like(hl) for _ in range(NDEV)]
    dist.all_gather(hs, hl)
    hf = torch.cat(hs, dim=2)
    del hs
    R = DIM // NDEV
    y = torch.zeros(M, K6, R, dtype=torch.float32, device=x.device)
    for e in ue:
        tk = (ids == e).nonzero()
        w2 = dq_fp4(w2h[e], s2h[e])
        y[tk[:, 0], tk[:, 1]] = bf16r(hf[tk[:, 0], tk[:, 1]] @ w2.t())
    acc = torch.zeros(M, R, dtype=torch.float32, device=x.device)
    for k in range(K6):
        acc = acc + y[:, k] * rw[:, k].float().unsqueeze(1)
    out = torch.zeros(M, DIM, dtype=torch.float32, device=x.device)
    out[:, R * rank:R * (rank + 1)] = bf16r(acc)
    return out


def combine_tp(locs):
    acc = locs[0].float().clone()
    for v in locs[1:]:
        acc = acc + v.float()
    return bf16r(acc)


def ref_dense_rank(w1, w2, x, ids, rw):
    """bf16 dense reference (gc-moe reference_moe_forward) restricted to this rank's slices, fp32, unrounded: summing the 8 ranks' partials
    gives the full layer exactly (gate / up rows and the FC2 K columns are partitioned)"""
    M = x.shape[0]
    out = torch.zeros(M, DIM, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in torch.unique(ids).tolist():
        if not (0 <= e < NEXP):
            continue
        rows, slots = (ids == e).nonzero(as_tuple=True)
        l1 = xf[rows] @ w1[e].float().t()
        mid = swiglu(l1[:, :INTER], l1[:, INTER:])
        out.index_add_(0, rows, rw[rows, slots].float().unsqueeze(1) * (mid @ w2[e].float().t()))
    return out


def stats(a, b):
    d = a.float() - b.float()
    return float(d.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-30))


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------ routings / inputs
def gc_tokens(T, rank, dev):
    """gc-moe build_tokens(T, ..., rank): x bf16 randn * 0.5, per-rank randn scores -> top-6 -> softmax"""
    g_x = torch.Generator(device=dev).manual_seed(42 + rank * 7919)
    x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev, generator=g_x) * 0.5
    g_r = torch.Generator(device=dev).manual_seed(1000 + rank * 31)
    scores = torch.randn(T, NEXP, dtype=torch.float32, device=dev, generator=g_r)
    w, idx = torch.topk(scores, TOPK, dim=-1, largest=True, sorted=False)
    return x, idx.to(torch.int32), torch.softmax(w, dim=-1)


def gc_batch(T, dev):
    """M = 8 T tokens: the concatenation of all 8 ranks' gc-moe tokens"""
    parts = [gc_tokens(T, r, dev) for r in range(NDEV)]
    return tuple(torch.cat([p[i] for p in parts]).contiguous() for i in range(3))


def skew_batch(M, dev, seed=0, sigma=1.5):
    """realistic skewed routing: per-expert popularity bias (N(0, sigma), fixed) + per-token N(0, 1) scores -> top-6 -> softmax of the scores"""
    g = torch.Generator(device=dev).manual_seed(5000 + seed)
    bias = torch.randn(NEXP, device=dev, generator=g) * sigma
    x = torch.randn(M, DIM, dtype=torch.bfloat16, device=dev, generator=g) * 0.5
    sc = torch.randn(M, NEXP, device=dev, generator=g) + bias
    w, idx = torch.topk(sc, TOPK, dim=-1)
    return x, idx.to(torch.int32).contiguous(), torch.softmax(w, dim=-1).contiguous()


# ------------------------------------------------------------------------------------------------ Zipf-skewed routing (all arms, same seeds)
ZIPF_SEED = 20261109


def zipf_popularity(s, dev):
    """per-expert popularity p_e ~ 1 / rank^s over a fixed random permutation of the 384 experts (same on every rank / arm)"""
    g = torch.Generator(device='cpu').manual_seed(ZIPF_SEED)
    perm = torch.randperm(NEXP, generator=g)
    p = torch.empty(NEXP, dtype=torch.float64)
    p[perm] = 1.0 / torch.arange(1, NEXP + 1, dtype=torch.float64) ** s
    return (p / p.sum()).float().to(dev)


def zipf_tokens(T, rank, dev, s):
    """rank's T tokens: x as gc_tokens; 6 distinct experts sampled by Zipf popularity (without replacement); weights = softmax of N(0, 1)"""
    g_x = torch.Generator(device=dev).manual_seed(42 + rank * 7919)
    x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev, generator=g_x) * 0.5
    g_r = torch.Generator(device=dev).manual_seed(ZIPF_SEED + 1 + rank * 31 + 100003 * T)
    p = zipf_popularity(s, dev)
    idx = torch.multinomial(p.expand(T, NEXP).contiguous(), TOPK, replacement=False, generator=g_r)
    w = torch.softmax(torch.randn(T, TOPK, device=dev, generator=g_r), dim=-1)
    return x, idx.to(torch.int32).contiguous(), w.contiguous()


REAL_PATH = '/home/scratch.aichenf_wwfo/v41_h20/cap_v41_l20.pt'   # (DSV4.1-Flash layer 20, B200 TP4 SGLang DSpark capture; v41_h20/cap41_build.py)
_REAL = {}


def real_rows(dev, path=None):
    """captured MoE inputs: x bf16 [R, 5120], production top-6 ids [R, 6], weights [R, 6] (routed scaling divided out, rows sum to 1)"""
    path = path or REAL_PATH
    k = (path, str(dev))
    if k not in _REAL:
        d = torch.load(path, map_location='cpu', weights_only=True)
        _REAL[k] = (d['x'].to(torch.bfloat16).to(dev).contiguous(), d['ids'].to(torch.int32).to(dev).contiguous(),
                    d['weights'].float().to(dev).contiguous())
    return _REAL[k]


DEC_DIR = '/home/scratch.aichenf_wwfo/routing_caps/dsv41_real'   # (DSV4.1-Flash plain-decode trace: tpmoe_real_headline.txt)
DEC_CASE = (20, 0)   # (layer, step) of --routing dec (set by the harness per case)
_DEC = {}


def dec_rows(M, dev, layer=None, step=None):
    """plain-decode trace rows of batch M at (layer, step): x bf16 [M, 5120] (the real layer-20 MoE input when (20, step) was captured, else
    seeded randn * 0.5), ids int32 [M, 6], weights fp32 [M, 6] (routed scaling 1.5 divided out: rows sum to 1)"""
    layer, step = (DEC_CASE if layer is None else (layer, step))
    key = (M, layer, step, str(dev))
    if key not in _DEC:
        import numpy as np
        d = np.load(f'{DEC_DIR}/dsv41_flash_routing_M{M:03d}.npz')
        li = int(np.nonzero(d['layer_ids'] == layer)[0][0])
        ids = torch.from_numpy(d['topk_ids'][step, li].astype(np.int32))
        w = torch.from_numpy(d['topk_weights'][step, li].astype(np.float32)) / 1.5
        xd = np.load(f'{DEC_DIR}/x/dsv41_flash_x_M{M:03d}.npz') if layer == 20 else None
        xs = xd['x_step_index'].astype(np.int64).tolist() if xd is not None else []
        if step in xs:   # (real layer-20 MoE input of exactly this step; headline v2: random x for timing, real x at L20 step 7 for checks)
            x = torch.from_numpy(xd['x_L20_bf16bits'][xs.index(step)].astype(np.int16)).view(torch.bfloat16)
        else:
            g = torch.Generator().manual_seed(77 + 1000 * layer + step + M)
            x = (torch.randn(M, DIM, generator=g) * 0.5).to(torch.bfloat16)
        _DEC[key] = (x.to(dev).contiguous(), ids.to(dev).contiguous(), w.to(dev).contiguous())
    return _DEC[key]


def route_tokens(T, rank, dev, routing='uniform', s=1.0):
    if routing == 'dec':   # (rank r's T tokens = rows r T .. r T + T - 1 of the M = 8 T batch)
        return tuple(v[rank * T:(rank + 1) * T].contiguous() for v in dec_rows(8 * T, dev))
    if routing == 'real':   # (rank r's T tokens = rows r T .. r T + T - 1, so the 8 ranks' concatenation = the first 8 T rows)
        return tuple(v[rank * T:(rank + 1) * T].contiguous() for v in real_rows(dev))
    return gc_tokens(T, rank, dev) if routing == 'uniform' else zipf_tokens(T, rank, dev, s)


def route_batch(T, dev, routing='uniform', s=1.0):
    """M = 8 T tokens: the concatenation of all 8 ranks' tokens (TP input)"""
    parts = [route_tokens(T, r, dev, routing, s) for r in range(NDEV)]
    return tuple(torch.cat([p[i] for p in parts]).contiguous() for i in range(3))


def ep_stats(ids, nrank=NDEV):
    """EP view of a TP batch: per-rank received routes (experts e on rank e // (NEXP / nrank)) max / mean, max expert load, unique experts"""
    r = (ids.long() // (NEXP // nrank)).flatten()
    cnt = torch.bincount(r, minlength=nrank).float()
    ecnt = torch.bincount(ids.long().flatten(), minlength=NEXP)
    return float(cnt.max() / cnt.mean()), int(ecnt.max()), int((ecnt > 0).sum())
