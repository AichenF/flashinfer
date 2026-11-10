"""suite_b200 common host code: cell shapes, synthetic MXFP4 TP slices, torch references, routings (real traces / uniform),
EP statistics, IPC helpers, the EP input quantizer (one Triton kernel) and the timing protocols.

Cells: DSV4-Flash (H4096 I2048 E256 top-6) and DSV4.1-Flash (H5120 I2304 E384 top-6), TP4 / TP8 on 8x B200.
TP slice of rank r: w13 rows [gate r I_r .. | up r I_r ..] (I_r = I / N), w2 columns r I_r ..; EP rank r holds experts r E/N ..
"""
import os
import statistics
import sys

import torch

SHAPES = {'dsv4f': dict(DIM=4096, INTER_FULL=2048, NEXP=256, TOPK=6),
          'dsv41': dict(DIM=5120, INTER_FULL=2304, NEXP=384, TOPK=6)}
REAL = {'dsv4f': '/home/scratch.aichenf_wwfo/suite_b200/data/cap_dsv4f_l20.pt',   # (DSV4-Flash L20 serving capture, 294 rows, = cmp3 cap_l20)
        'dsv41': '/home/scratch.aichenf_wwfo/v41_h20/cap_v41_l20.pt'}             # (DSV4.1-Flash L20 B200 TP4 DSpark capture, 732 rows)
CLAMP = 10.0


class Shape:
    def __init__(self, model, ndev):
        self.model, self.ndev = model, ndev
        for k, v in SHAPES[model].items():
            setattr(self, k, v)
        self.INTER = self.INTER_FULL // ndev


# ------------------------------------------------------------------------------------------------ weights
def synth_slices(sh, rank, dev, seed=7, scale=0.05):
    """this rank's bf16 TP slices: w1 [E][2 I_r][DIM] = [gate rows; up rows], w2 [E][DIM][I_r]"""
    g = torch.Generator(device=dev).manual_seed(seed + 1000 * rank + 17 * sh.ndev)
    w1 = torch.randn(sh.NEXP, 2 * sh.INTER, sh.DIM, dtype=torch.bfloat16, device=dev, generator=g) * scale
    w2 = torch.randn(sh.NEXP, sh.DIM, sh.INTER, dtype=torch.bfloat16, device=dev, generator=g) * scale
    return w1, w2


_E2M1 = {}


def _e2m1_table(dev):
    if dev not in _E2M1:
        _E2M1[dev] = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=dev)
    return _E2M1[dev]


def quant_fp4(x):
    """MXFP4 (DeepGEMM per_token_cast_to_fp4(use_ue8m0=True, gran_k=32)): rows [R, K] -> packed uint8 [R, K/2] (low nibble = even
    element), E8M0 bytes [R, K/32]"""
    from deep_gemm.utils import per_token_cast_to_fp4
    packed, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
    e8 = (torch.log2(sf.float()).round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    return packed.view(torch.uint8).contiguous(), e8.contiguous()


def dq_fp4(packed, e8):
    R = packed.shape[0]
    v = torch.stack([packed & 15, packed >> 4], -1).reshape(R, -1).long()
    return _e2m1_table(packed.device)[v] * torch.exp2(e8.float() - 127.0).repeat_interleave(32, dim=1)


def quantize_slices(sh, w1, w2, chunk=16):
    """bf16 slices -> MXFP4 raw (source row order): w13 [E][2 I_r][DIM/2] + s13 [E][2 I_r][DIM/32]; w2 [E][DIM][I_r/2] + s2 [E][DIM][I_r/32]"""
    E, I = w1.shape[0], sh.INTER
    q1, s1, q2, s2 = [], [], [], []
    for e0 in range(0, E, chunk):
        a, b = quant_fp4(w1[e0:e0 + chunk].reshape(-1, sh.DIM))
        q1.append(a.view(-1, 2 * I, sh.DIM // 2)); s1.append(b.view(-1, 2 * I, sh.DIM // 32))
        a, b = quant_fp4(w2[e0:e0 + chunk].reshape(-1, I))
        q2.append(a.view(-1, sh.DIM, I // 2)); s2.append(b.view(-1, sh.DIM, I // 32))
    return dict(w13=torch.cat(q1), s13=torch.cat(s1), w2=torch.cat(q2), s2=torch.cat(s2))


# ------------------------------------------------------------------------------------------------ references
def e4m3(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


def _fbits(x):
    return x.contiguous().view(torch.int32)


def mxfp8_x(x):
    """x fp32 [M, K] -> dequantized MXFP8, scale 2^ceil(log2(amax / 448)) (exact bit rule; smallest normal for 0)"""
    M, K = x.shape
    xb = x.view(M, K // 32, 32)
    am = xb.abs().amax(-1) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    am = torch.maximum(am, torch.tensor(1.17549435e-38, device=x.device))
    b = _fbits(am)
    ex = ((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0).int()
    s = torch.exp2(ex.float()).unsqueeze(-1)
    return (e4m3(xb / s) * s).view(M, K)


def mxfp8_h(a):
    """FC1 activation -> dequantized MXFP8 with the OCP rule 2^(floor(log2 amax) - 8) per 32"""
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


def ref_rank(sh, raw, x, ids, rw, bf16_weights=False):
    """one rank's partial with the kernels' numerics: bf16(sum_k w_k bf16(FC2(MXFP8h(SwiGLU(FC1(MXFP8x(x))))))), slots in order"""
    M, I = x.shape[0], sh.INTER
    xq = mxfp8_x(x.float())
    y = torch.zeros(M, ids.shape[1], sh.DIM, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not (0 <= e < sh.NEXP):
            continue
        tk = (ids == e).nonzero()
        w1 = dq_fp4(raw['w13'][e], raw['s13'][e])
        w2 = dq_fp4(raw['w2'][e], raw['s2'][e])
        gu = xq[tk[:, 0]] @ w1.t()
        h = mxfp8_h(swiglu(gu[:, :I], gu[:, I:]))
        y[tk[:, 0], tk[:, 1]] = bf16r(h @ w2.t())
    w = bf16r(rw.float()) if bf16_weights else rw.float()
    acc = torch.zeros(M, sh.DIM, dtype=torch.float32, device=x.device)
    for k in range(ids.shape[1]):
        acc = acc + y[:, k] * w[:, k].unsqueeze(1)
    return bf16r(acc)


def combine_tp(locs):
    acc = locs[0].float().clone()
    for v in locs[1:]:
        acc = acc + v.float()
    return bf16r(acc)


def ref_dense_rank(sh, w1, w2, x, ids, rw):
    """bf16-weight dense reference restricted to this rank's slices (fp32 math, unrounded): the N ranks' partials sum to the layer"""
    M, I = x.shape[0], sh.INTER
    out = torch.zeros(M, sh.DIM, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in torch.unique(ids).tolist():
        if not (0 <= e < sh.NEXP):
            continue
        rows, slots = (ids == e).nonzero(as_tuple=True)
        l1 = xf[rows] @ w1[e].float().t()
        mid = swiglu(l1[:, :I], l1[:, I:])
        out.index_add_(0, rows, rw[rows, slots].float().unsqueeze(1) * (mid @ w2[e].float().t()))
    return out


def rel(a, b):
    d = a.float() - b.float()
    return float(d.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-30))


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------ routings
_REALC = {}


def real_rows(model, dev):
    """captured MoE inputs: x bf16 [R, DIM], production top-6 ids [R, 6], weights fp32 [R, 6] (rows sum to 1)"""
    k = (model, str(dev))
    if k not in _REALC:
        d = torch.load(REAL[model], map_location='cpu', weights_only=False)
        w = d['weights'].float()   # (rows sum to 1: routed scaling already divided out)
        _REALC[k] = (d['x'].to(torch.bfloat16).to(dev).contiguous(), d['ids'].to(torch.int32).to(dev).contiguous(), w.to(dev).contiguous())
    return _REALC[k]


def uniform_tokens(sh, n, seed, dev):
    """gc-moe build_tokens style: x bf16 randn * 0.5, scores randn -> top-6 -> softmax"""
    g_x = torch.Generator(device=dev).manual_seed(42 + seed * 7919)
    x = torch.randn(n, sh.DIM, dtype=torch.bfloat16, device=dev, generator=g_x) * 0.5
    g_r = torch.Generator(device=dev).manual_seed(1000 + seed * 31)
    scores = torch.randn(n, sh.NEXP, dtype=torch.float32, device=dev, generator=g_r)
    w, idx = torch.topk(scores, sh.TOPK, dim=-1, largest=True, sorted=False)
    return x, idx.to(torch.int32).contiguous(), torch.softmax(w, dim=-1).contiguous()


def tp_batch(sh, M, dev, routing):
    """the M tokens every TP rank sees = the concatenation of the N EP ranks' T = M / N tokens (real: the first M trace rows)"""
    if routing == 'real':
        x, ids, w = real_rows(sh.model, dev)
        assert M <= x.shape[0], f'real trace has {x.shape[0]} rows < M = {M}'
        return x[:M].contiguous(), ids[:M].contiguous(), w[:M].contiguous()
    T = M // sh.ndev
    parts = [uniform_tokens(sh, T, r + 1000 * M, dev) for r in range(sh.ndev)]
    return tuple(torch.cat([p[i] for p in parts]).contiguous() for i in range(3))


def ep_tokens(sh, M, rank, dev, routing):
    """EP rank r's T = M / N tokens: rows r T .. r T + T - 1 of tp_batch"""
    T = M // sh.ndev
    return tuple(v[rank * T:(rank + 1) * T].contiguous() for v in tp_batch(sh, M, dev, routing))


TRACES = {'dsv41': '/home/scratch.aichenf_wwfo/routing_caps/dsv41_real', 'dsv4f': '/home/scratch.aichenf_wwfo/routing_caps/dsv4f_real'}
SUITE = '/home/scratch.aichenf_wwfo/tpmoe_suite'


def trace_cases(sh, M, dev):
    """plain-decode real-trace headline (tpmoe_real_headline.txt v2/v3) through the SUITE's own loader (traces/loaders.py Routing,
    'real-trace:<dir>'): timing cases L10/20/30 x steps 32/96 with random bf16 x (loaders.synth_x(M, H, case index)), weights / 1.5;
    + its real-x correctness case (layer 20, steady step 33)"""
    if SUITE + '/traces' not in sys.path:
        sys.path.insert(0, SUITE + '/traces')
    import loaders
    R = loaders.Routing('real-trace:' + TRACES[sh.model], (sh.DIM, sh.INTER_FULL, sh.NEXP, sh.TOPK))
    out = []
    for ci, (lab, x, ids, w) in enumerate(R.cases(M)):
        x = loaders.synth_x(M, sh.DIM, ci) if x is None else x
        out.append((lab, x.contiguous().to(dev), ids.to(torch.int32).contiguous().to(dev), w.float().contiguous().to(dev)))
    lab, x, ids, w = R.check_case(M)
    return out, (lab, x.contiguous().to(dev), ids.to(torch.int32).contiguous().to(dev), w.float().contiguous().to(dev))


def tp_cases(sh, M, dev, routing):
    """(timing cases [(label, x, ids, w)], the correctness case) of M: one case = itself for uniform / the L20 capture; 6 + 1 for 'trace'"""
    if routing == 'trace':
        return trace_cases(sh, M, dev)
    c = (routing,) + tuple(tp_batch(sh, M, dev, routing))
    return [c], c


def ep_stats(sh, ids):
    """EP view of a TP batch: per-rank received routes (expert e on rank e // (E / N)) max / mean, max expert load, unique experts"""
    r = (ids.long() // (sh.NEXP // sh.ndev)).flatten()
    cnt = torch.bincount(r, minlength=sh.ndev).float()
    ecnt = torch.bincount(ids.long().flatten(), minlength=sh.NEXP)
    return float(cnt.max() / cnt.mean()), int(ecnt.max()), int((ecnt > 0).sum())


# ------------------------------------------------------------------------------------------------ EP input quantizer (one kernel)
_EPQ = None


def ep_quant_kernel():
    """x bf16 [T, H] -> e4m3 [T, H] + packed UE8M0 int32 [T, H / 128] (4 per-32 scales per int, low byte first), scale
    2^ceil(log2(max(amax, 1e-4) / 448)) (DeepGEMM per_token_cast_to_fp8(use_ue8m0=True, gran_k=32, use_packed_ue8m0=True) rule)"""
    global _EPQ
    if _EPQ is not None:
        return _EPQ
    import triton
    import triton.language as tl

    @triton.jit
    def _k(x_ptr, q_ptr, s_ptr, H: tl.constexpr):
        t = tl.program_id(0)
        b = tl.program_id(1)
        offs = b * 128 + tl.arange(0, 128)
        x = tl.load(x_ptr + t * H + offs).to(tl.float32)
        xg = tl.reshape(x, (4, 32))
        am = tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-4)
        v = tl.math.div_rn(am, 448.0)
        bits = v.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        inv = ((254 - e) << 23).to(tl.float32, bitcast=True)   # 1 / 2^(e - 127)
        q = (xg * inv[:, None]).to(tl.float8e4nv)
        tl.store(q_ptr + t * H + offs, tl.reshape(q, (128,)))
        sh_ = tl.arange(0, 4) * 8
        packed = tl.sum(e << sh_, axis=0)
        tl.store(s_ptr + t * (H // 128) + b, packed)

    def run(x, q, s):
        T, H = x.shape
        _k[(T, H // 128)](x, q, s, H=H)
    _EPQ = run
    return run


# ------------------------------------------------------------------------------------------------ IPC
class CudaRT:
    def __init__(self):
        import ctypes, glob
        cands = glob.glob(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libcudart*.so*'))
        cands += glob.glob(os.path.join(sys.prefix, 'lib', 'python*', 'site-packages', 'nvidia', '**', 'libcudart.so*'), recursive=True)
        cands += ['/usr/local/cuda/lib64/libcudart.so']
        self.ct = ctypes
        self.lib = ctypes.CDLL(cands[0])

        class IpcHandle(ctypes.Structure):
            _fields_ = [('reserved', ctypes.c_char * 64)]
        self.IpcHandle = IpcHandle
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcHandle), ctypes.c_void_p]
        self.lib.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), IpcHandle, ctypes.c_uint]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def ck(self, r):
        assert r == 0, self.lib.cudaGetErrorString(r)

    def malloc(self, n):
        p = self.ct.c_void_p(); self.ck(self.lib.cudaMalloc(self.ct.byref(p), n)); self.ck(self.lib.cudaMemset(p, 0, n)); return p.value

    def handle(self, ptr):
        h = self.IpcHandle(); self.ck(self.lib.cudaIpcGetMemHandle(self.ct.byref(h), self.ct.c_void_p(ptr))); return bytes(h)

    def open(self, raw):
        p = self.ct.c_void_p(); self.ck(self.lib.cudaIpcOpenMemHandle(self.ct.byref(p), self.IpcHandle.from_buffer_copy(raw), 1)); return p.value


def ipc_table(rt, nbytes, rank, ndev, gloo):
    """cudaMalloc nbytes (zeroed) on this rank, exchange IPC handles: [peer pointers] (own pointer at index rank)"""
    import torch.distributed as dist
    ptr = rt.malloc(nbytes)
    hs = [None] * ndev
    dist.all_gather_object(hs, rt.handle(ptr), group=gloo)
    return [ptr if p == rank else rt.open(h) for p, h in enumerate(hs)]


# ------------------------------------------------------------------------------------------------ timing
class Timer:
    """protocol b (headline): graph of 1 call, 512 MiB L2 flush + GPU rank alignment (tiny NCCL all-reduce + ~100 us GPU sleep) before
    EVERY replay, per-rank median over cold_iters, max over ranks.  protocol a (gc-moe's): 100 replays per window (L2 warm across
    replays), flush between windows, 10 warm-up + `windows` windows, per-rank median, max over ranks."""

    def __init__(self, dev, gloo, ndev, cold_iters=50, windows=30, inner=100, sleep_cycles=200000):
        import torch.distributed as dist
        self.dist, self.gloo, self.ndev = dist, gloo, ndev
        self.tiny = torch.zeros(1, device=dev)
        self.flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
        self.cold_iters, self.windows, self.inner, self.sleep_cycles = cold_iters, windows, inner, sleep_cycles

    def align(self):
        self.dist.all_reduce(self.tiny)
        torch.cuda._sleep(self.sleep_cycles)

    def capture(self, fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); self.dist.barrier(group=self.gloo)
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize(); self.dist.barrier(group=self.gloo)
        return g

    def time(self, g, proto):
        res = []
        if proto == 'a':
            for wi in range(10 + self.windows):
                self.dist.barrier(group=self.gloo)
                self.flush.zero_()
                self.align()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(self.inner):
                    g.replay()
                e1.record(); torch.cuda.synchronize()
                if wi >= 10:
                    res.append(e0.elapsed_time(e1) * 1e3 / self.inner)
        else:
            for wi in range(3 + self.cold_iters):
                if wi % 25 == 0:
                    self.dist.barrier(group=self.gloo)
                self.flush.zero_()
                self.align()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record()
                torch.cuda.synchronize()
                if wi >= 3:
                    res.append(e0.elapsed_time(e1) * 1e3)
        srt = sorted(res); k = len(srt) // 5
        med, mean = statistics.median(res), statistics.fmean(srt[k:len(srt) - k] or srt)   # (mean = 20%-trimmed mean)
        allm = [None] * self.ndev
        self.dist.all_gather_object(allm, (med, mean), group=self.gloo)
        self.last_mean = max(v[1] for v in allm)   # (B200 event timestamps are quantized to ~2 us: the mean resolves sub-quantum A/B differences)
        return max(v[0] for v in allm), [v[0] for v in allm]
