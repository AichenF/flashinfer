"""Host side of the B300 TP-MoE suite kernel s3.cu (port_s3.py): shape selection, build, per-rank weight slices in the kernel's layouts,
workspace, launch wrapper, torch references (the kernel's numerics; bf16 dense like gc-moe's reference_moe_forward), routings.

Shape from the environment (read at import, inherited by spawned ranks): S3_MODEL = v4f (DeepSeek-V4-Flash: hidden 4096, I 2048, 256
experts) | v41 (DeepSeek-V4.1-Flash: hidden 5120, I 2304, 384 experts); S3_TP = 4 | 8 (TP size = world size = EP size of the baselines).
Derived from ../tp8_v41_b300/v41host.py (agent B300-TP8)."""
import os
import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_EXT = {}
MODEL = os.environ.get('S3_MODEL', 'v4f')
NDEV = int(os.environ.get('S3_TP', '8'))
SHAPES = {'v4f': (4096, 2048, 256), 'v41': (5120, 2304, 384)}
DIM, INTER_FULL, NEXP = SHAPES[MODEL]
TOPK = 6
assert NDEV in (4, 8)
INTER = INTER_FULL // NDEV            # per-rank intermediate: v4f 512 / 256, v41 576 / 288
NT1 = 2 * INTER // 128                # FC1 128-row tiles per expert
NCH1 = DIM // 128
NRT2 = DIM // 128                     # FC2 128-row tiles
NCH2 = (INTER + 127) // 128           # FC2 K chunks (v41 TP4: 4.5 -> 5)
CLAMP = 10.0
assert (2 * INTER) % 128 == 0, 'v41 TP8 (I 288) is the B300-TP8 agent\'s kernel (tp8v41.cu)'
KSRC = os.environ.get('S3_KSRC', 's3')   # kernel source: s3 (port_s3.py over k3h.cu) | v4ft (port_v4f.py over B300-TP8's tp8v41.cu)
assert KSRC in ('s3', 'v4ft', 'v4fth', 'v4fti') and (KSRC == 's3' or MODEL == 'v4f')   # (v4fth: v4ft on a newer tp8v41 with H2 work; v4fti: on tp8v41 HEAD, B300-TP8 11-10)


def nmax32_minm(ksrc=None):   # > 0: builds with MMIN >= this use NMAX 32 (tp8v41's per-M rule; v4ft default 256, s3 off)
    ksrc = ksrc or KSRC
    return int(os.environ.get('S3_NMAX32_MINM', '256' if ksrc.startswith('v4ft') else '0'))


WS_NMAX = 32   # (workspace sized for the largest panel width of any build)
# (v4ft) LLS sentinel all-reduce (B300-TP8 / suite_b200 tail knob; the host 0xFF-fills the RS buffer's LL region, LLS and non-LLS builds never
# share a workspace): B300b v10, 2 reps swapped, M8..512 TP8 -0/-0/-1/-1/-2/-4/-5/-8 us, TP4 -0/-0/-0/-1/-3/-4/-6/-7 us; ranks bitwise, b2b + b2b-mix 0 bad
V4FT_COMMON = ('-DTPB2_LLS=1',)


def f2h_minm(ksrc=None, ndev=None):   # ((v41 H2) F2H, v4ft only) > 0: builds with MMIN >= this shard FC2 on H (h exchange, local T1, all-gather)
    if not (ksrc or KSRC).startswith('v4ft') or (ndev or NDEV) == 1:
        return 0
    return int(os.environ.get('S3_F2H_MINM', '0'))


def shape_flags():
    return (f'-DS3_DIM={DIM}', f'-DS3_IFULL={INTER_FULL}', f'-DS3_NEXP={NEXP}', f'-DS3_NDEV={NDEV}')


def load_ext(src=None, tag='', flags=(), verbose=False):
    src = src or f'{KSRC}.cu'
    tag = re.sub(r'[^A-Za-z0-9_]', '_', f'{MODEL}t{NDEV}_{tag}')
    flags = shape_flags() + tuple(flags) + tuple(os.environ.get('S3_FLAGS', '').split())
    key = (src, tag, flags)
    if key in _EXT:
        return _EXT[key]
    import hashlib
    tag = f'{tag}_{hashlib.md5((" ".join(flags) + src).encode()).hexdigest()[:8]}'
    if len(tag) > 120:
        tag = tag[:60] + hashlib.md5(tag.encode()).hexdigest()[:12]
    from torch.utils.cpp_extension import load
    maj, mnr = torch.cuda.get_device_capability()
    arch = f'{maj}{mnr}a'
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{maj}.{mnr}a'
    build = Path(os.environ.get('S3_BUILD', str(HERE / 'build'))) / f'b_{tag}'
    build.mkdir(parents=True, exist_ok=True)
    name = f's3{tag}'
    m = load(name=name, sources=[str(HERE / src)], build_directory=str(build),
             extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
                                f'-DTPB2_NS=ns_{tag}', *flags],
             extra_include_paths=[str(HERE)], extra_ldflags=['-lcuda'], verbose=verbose)
    _EXT[key] = m
    return m


SPEC_COMMON = ('-DTPB2_SPEC=1', '-DTPB2_WHOLE=1', '-DTPB2_RS=0', '-DTPB2_QX1=1', '-DTPB2_PUT1=1', '-DTPB2_CEPOCH=1',
               '-DTPB2_NMAX=24', '-DTPB2_NS_STAGES=5', '-DTPB2_EARLYW=2', '-DTPB2_IDV=1', '-DTPB2_XRLX=1', '-DTPB2_EPI2=1')


def spec_flags(mlo, mhi, ndev, grid=148, extra=(), ksrc=None):
    """compile-time specialized build for mlo <= M <= mhi at TP size ndev (1: rank-local test); an extra -DX=.. replaces a default -DX"""
    ex = tuple(extra)
    over = {f.split('=')[0] for f in ex if f.startswith('-D')}
    base = (f'-DTPB2_MMIN={mlo}', f'-DTPB2_MMAX={mhi}', f'-DTPB2_NDEV={ndev}', f'-DTPB2_GRID={grid}') + SPEC_COMMON
    if (ksrc or KSRC).startswith('v4ft'):
        base += V4FT_COMMON
    if 0 < nmax32_minm(ksrc) <= mlo:
        base = tuple('-DTPB2_NMAX=32' if f == '-DTPB2_NMAX=24' else f for f in base)
    if 0 < f2h_minm(ksrc, ndev) <= mlo:
        base += ('-DTPB2_F2H=1',)
    return tuple(f for f in base if f.split('=')[0] not in over) + ex


# ------------------------------------------------------------------------------------------------ weights
def synth_slices(rank, dev, seed=7, scale=0.05):
    """this rank's bf16 slices: w1 [E][2 I][DIM] = [gate rows; up rows] of its intermediate slice (gc-moe order), w2 [E][DIM][I]"""
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
    """bf16 slices -> MXFP4 raw (source row order): w13 [E][2 I][DIM / 2] + s13 [E][2 I][DIM / 32]; w2 [E][DIM][I / 2] + s2 [E][DIM][I / 32]"""
    E = w1.shape[0]
    q1, s1, q2, s2 = [], [], [], []
    for e0 in range(0, E, chunk):
        a, b = quant_fp4(w1[e0:e0 + chunk].reshape(-1, DIM))
        q1.append(a.view(-1, 2 * INTER, DIM // 2)); s1.append(b.view(-1, 2 * INTER, DIM // 32))
        a, b = quant_fp4(w2[e0:e0 + chunk].reshape(-1, INTER))
        q2.append(a.view(-1, DIM, INTER // 2)); s2.append(b.view(-1, DIM, INTER // 32))
    return dict(w13=torch.cat(q1), s13=torch.cat(s1), w2=torch.cat(q2), s2=torch.cat(s2))


def w13_perm(dev):
    """stored row s of the kernel's FC1 tile layout <- source row (gate 0..I-1 | up I..2I-1): 32-row blocks of interleaved (up_j, gate_j) pairs,
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
    """raw MXFP4 slices -> the kernel's tensors: w13 [E][2 I][DIM / 2] (gated interleave + 32-row shuffle) + SF images [E][NT1][NCH1][512];
    w2 [E][DIM][I / 2] natural rows (+ 64 B tail slack: the last chunk of the last row of a ragged I reads past it) + SF images
    [E][NRT2][NCH2][512] (ragged I: missing K blocks scale 2^0, never used)"""
    E, dev = raw['w13'].shape[0], raw['w13'].device
    perm = w13_perm(dev)
    w13 = raw['w13'][:, perm].contiguous()
    s13 = _sf_images(raw['s13'][:, perm], NT1, NCH1)
    w2f = torch.zeros(raw['w2'].numel() + 4096, dtype=torch.uint8, device=dev)
    w2f[:raw['w2'].numel()] = raw['w2'].reshape(-1)
    w2 = w2f[:raw['w2'].numel()].view(E, DIM, INTER // 2)
    s2r = raw['s2']
    if 4 * NCH2 > s2r.shape[2]:
        s2r = torch.cat([s2r, torch.full((E, DIM, 4 * NCH2 - s2r.shape[2]), 127, dtype=torch.uint8, device=dev)], 2)
    s2 = _sf_images(s2r.contiguous(), NRT2, NCH2)
    return dict(w13=w13, s13=s13.view(E, -1), w2=w2, s2=s2.view(E, -1), w2_store=w2f, lanes=torch.arange(128, dtype=torch.uint8, device=dev))


class Weights:
    def __init__(self, ext, kl):
        self.kl = kl
        self.w13, self.w2 = kl['w13'], kl['w2']
        self.w13_sf, self.w2_sf, self.lanes = kl['s13'], kl['s2'], kl['lanes']
        self.maps = ext.encode_maps(self.w13, self.w2)
        if int(getattr(ext, 'F2H', 0)):   # ((v41 H2) F2H) this rank's H-sharded FC2 weights: kl['f2h'] = f2h_weights(..), else synthetic (timing only)
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
        self.xg = torch.zeros(E.MAXP * E.NCH1 * max(E.NMAX, WS_NMAX) * 128, **u8)   # (GATHER=0 builds: per-panel x images; one workspace serves every build)
        self.xgsf = torch.zeros(E.MAXP * E.NSTG1 * 512, **u8)
        self.himg = torch.zeros(E.MAXP * E.NCH2 * max(E.NMAX, WS_NMAX) * 128 + 4096, **u8)
        self.hsf = torch.zeros(E.MAXP * max(E.NSTG2, 4) * 512 + 4096, **u8)
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
        self.grid = int(os.environ.get('S3_GRID', 0)) or torch.cuda.get_device_properties(dev).multi_processor_count

    @staticmethod
    def RS_BYTES(E):   # (+ TB_BYTES: the suite_b200 tail stage's TAILB counters live past the LL halves; v4ft zeroes them only in TAILB builds)
        return 2 * int(E.RS_PAR) * 16 + (1 << 20) + int(getattr(E, 'TB_BYTES', 0))

    def link_ptrs(self, ptrs):
        self.rs = [int(p) for p in ptrs]


def run(st, W, x, ids, rw, out, dbg=None, E=None):
    """out [M, DIM] bf16 = this rank's routed sum (ndev 1) or the TP all-reduce of it.  ids int32 [M][6], rw fp32 [M][6]."""
    E = E or st.E
    if int(getattr(E, 'F2H', 0)):   # ((v41 H2) F2H) gathered-h buffers: st.hgp (TP: symmetric, set by the harness), else one local buffer
        if getattr(st, 'hgp', None) is None:
            st.hg = torch.zeros(int(E.HG_BYTES) * 2, dtype=torch.uint8, device=st.dev)
            st.hgp = [st.hg.data_ptr()]
        key = (id(W), id(st), tuple(st.hgp))
        if getattr(E, '_f2h_key', None) != key:   # (the build's tensor maps: encoded once per (weights, state))
            E.set_f2h(W.w2h, W.w2h_sf, st.hgp, st.rank)
            E._f2h_key = key
    rs = st.rs if st.ndev > 1 else []
    E.run(W.maps, W.w13_sf, W.w2_sf, W.lanes, x, None, None, ids, rw, None, 1.0, st.xq, st.xs, st.xg, st.xgsf, dbg, st.himg, st.hsf,
          st.part, st.part_flags, st.xflag, st.gcnt, st.pan_cnt, st.rdy, st.fc2_next, st.rt_cnt, st.work, st.outp, rs, rs, out,
          st.rank, st.ndev, st.grid, None, None, None, st.small, st.small, st.small, st.small, st.shout, st.xsf2, st.tcnt,
          [W.w13.data_ptr(), W.w2.data_ptr(), 0, 0], None, None, None, 1e-5)
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
    """((v41 H2) F2H) this rank's H-sharded FC2 weights from every rank's canonical MXFP4 TP slices (all-gathered, collective): rows
    DIM / NDEV rank .. of the full quantized W2 (K slices concatenated: per-32 scales, so the full-K quantization IS the slices') -> fp4
    [E][DIM / NDEV][I_full / 2], E8M0 [E][DIM / NDEV][I_full / 32]  (from tp8_v41_b300/v41host.py)"""
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
    """((v41 H2) F2H) one rank's partial with the H2 numerics (collective): h slices (as ref_rank) all-gathered -> full K, FC2 of this
    rank's DIM / NDEV rows (fp32, one bf16 rounding per slot), slots in order (fp32) -> bf16 in rows R rank .., 0 elsewhere"""
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
    """bf16 dense reference (gc-moe reference_moe_forward) restricted to this rank's slices, fp32, unrounded: summing the ranks' partials
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


def skew_batch(M, dev, seed=0, sigma=1.5):
    """synthetic skewed routing: per-expert popularity bias (N(0, sigma), fixed) + per-token N(0, 1) scores -> top-6 -> softmax of the scores"""
    g = torch.Generator(device=dev).manual_seed(5000 + seed)
    bias = torch.randn(NEXP, device=dev, generator=g) * sigma
    x = torch.randn(M, DIM, dtype=torch.bfloat16, device=dev, generator=g) * 0.5
    sc = torch.randn(M, NEXP, device=dev, generator=g) + bias
    w, idx = torch.topk(sc, TOPK, dim=-1)
    return x, idx.to(torch.int32).contiguous(), torch.softmax(w, dim=-1).contiguous()


# real layer-20 serving captures: DSV4.1-Flash (v41_h20/cap41_build.py, 732 rows) | DSV4-Flash (cap_v4f_build.py: the B200/H100 serving
# capture's layer-20 hidden states c8 + c4 + c1 routed by the checkpoint's layer-20 router, 294 rows; == cmp3's cap_l20.pt statistics)
REAL_PATHS = {'v41': '/home/scratch.aichenf_wwfo/v41_h20/cap_v41_l20.pt',
              'v4f': '/home/scratch.aichenf_wwfo/tpmoe_suite/traces/data/dsv4f_cap_l20.pt'}
_REAL = {}


def real_rows(dev, path=None):
    """captured MoE inputs: x bf16 [R, DIM], production top-6 ids [R, 6], weights [R, 6] (routed scaling divided out, rows sum to 1)"""
    path = path or os.environ.get('S3_REAL') or REAL_PATHS[MODEL]
    k = (path, str(dev))
    if k not in _REAL:
        d = torch.load(path, map_location='cpu', weights_only=True)
        _REAL[k] = (d['x'].to(torch.bfloat16).to(dev).contiguous(), d['ids'].to(torch.int32).to(dev).contiguous(),
                    d['weights'].float().to(dev).contiguous())
    return _REAL[k]


def real_count(dev):
    return real_rows(dev)[0].shape[0]


TRACE_DIRS = {'v41': '/home/scratch.aichenf_wwfo/routing_caps/dsv41_real', 'v4f': '/home/scratch.aichenf_wwfo/routing_caps/dsv4f_real'}
_TRACE = {}


def trace_rows(M, dev, case=None):
    """plain-decode trace (routing_caps/<model>_real): the M-token batch of layer L, step S (S3_TRACE_CASE='L,S', default 20,32); x = seeded
    randn * 0.5 (the suite's timing cases use random x); weights / their row sum (routed scaling 1.5 removed)"""
    import glob
    import numpy as np
    L, S = (int(v) for v in (case or os.environ.get('S3_TRACE_CASE', '20,32')).split(','))
    k = (M, L, S, str(dev))
    if k not in _TRACE:
        f = sorted(glob.glob(f'{TRACE_DIRS[MODEL]}/*_routing_M{M:03d}.npz'))[0]
        z = np.load(f)
        li = [int(v) for v in z['layer_ids']].index(L)
        ids = torch.from_numpy(z['topk_ids'][S, li].astype('int64')).to(torch.int32)
        w = torch.from_numpy(z['topk_weights'][S, li].astype('float32'))
        w = w / w.sum(-1, keepdim=True)
        g = torch.Generator().manual_seed(4242 + 7919 * L + M)
        x = (torch.randn(M, DIM, generator=g) * 0.5).bfloat16()
        _TRACE[k] = (x.to(dev).contiguous(), ids.to(dev).contiguous(), w.to(dev).contiguous())
    return _TRACE[k]


def route_tokens(T, rank, dev, routing='uniform'):
    if routing == 'trace':
        return tuple(v[rank * T:(rank + 1) * T].contiguous() for v in trace_rows(T * NDEV, dev))
    if routing == 'real':   # (rank r's T tokens = rows r T .. r T + T - 1, so the N ranks' concatenation = the first N T rows)
        return tuple(v[rank * T:(rank + 1) * T].contiguous() for v in real_rows(dev))
    return gc_tokens(T, rank, dev)


def route_batch(T, dev, routing='uniform'):
    """M = N T tokens: the concatenation of all N ranks' tokens (TP input)"""
    parts = [route_tokens(T, r, dev, routing) for r in range(NDEV)]
    return tuple(torch.cat([p[i] for p in parts]).contiguous() for i in range(3))


def ep_stats(ids, nrank=NDEV):
    """EP view of a TP batch: per-rank received routes (experts e on rank e // (NEXP / nrank)) max / mean, max expert load, unique experts"""
    r = (ids.long() // (NEXP // nrank)).flatten()
    cnt = torch.bincount(r, minlength=nrank).float()
    ecnt = torch.bincount(ids.long().flatten(), minlength=NEXP)
    return float(cnt.max() / cnt.mean()), int(ecnt.max()), int((ecnt > 0).sum())
