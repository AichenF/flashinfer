"""fp32 torch reference of the B200 production MoE numerics (flashinfer_mxfp4 path), rank-local, before the all-reduce.

routed(t) = bf16( sum_k bf16(w_tk) * FC2_e( q(act) ) ),  act = up * gate * sigmoid(gate) with gate <= 10, |up| <= 10,
[gate | up] = FC1_e(q(x_t)), q() = MXFP8 (e4m3, one UE8M0 scale per 32: 2^ceil(log2(amax / 448)), RNE saturating),
MXFP4 weights dequantized exactly (e2m1 * 2^(e - 127)).  All accumulation in fp32 (the reference has no K order).
Measured against trtllm-gen (b2_refcheck): x quantized with the ceil rule (bitwise == flashinfer mxfp8_quantize), h with the OCP
floor rule, routing weights after FC2, gate / up / act in fp32 -> rel 0.0026 (vs 0.025 with a ceil-rule h)."""
import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant_mxfp4(w, s):
    """w int8/uint8 [..., N, K/2] (element 2j in the low nibble), s fp32 [..., N, K/32] (power-of-two values) -> fp32 [..., N, K]"""
    lut = torch.tensor(E2M1, dtype=torch.float32, device=w.device)
    u = w.view(torch.uint8).to(torch.int64)
    v = torch.stack([lut[u & 15], lut[u >> 4]], dim=-1).flatten(-2)
    return v * s.float().repeat_interleave(32, dim=-1)


def quant_mxfp8(x, block=32, rule='ceil'):
    """-> (dequantized fp32 values, scale exponents int [.., K/block]), RNE saturating e4m3.  rule 'ceil': scale 2^ceil(log2(amax/448))
    (flashinfer mxfp8_quantize: the MoE input x); 'floor': 2^(floor(log2 amax) - 8) (OCP MX; trtllm-gen's FC1 output h, measured)"""
    sh = x.shape
    xb = x.float().reshape(*sh[:-1], sh[-1] // block, block)
    amax = xb.abs().amax(-1, keepdim=True)
    if rule == 'floor':
        e = (torch.floor(torch.log2(amax.clamp_min(1e-38))) - 8).clamp(-127, 127)
    else:
        e = torch.ceil(torch.log2(amax.clamp_min(1e-38) / 448.0)).clamp(-127, 127)
    s = torch.exp2(e)
    q = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(sh), e.squeeze(-1).to(torch.int32)


def swiglu_clamp(gate, up, limit=10.0):
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return up * gate * torch.sigmoid(gate)


def routed_ref(x, ids, w, raw, xq=None, act_bf16=False):
    """x bf16 [M, DIM]; ids int [M, 6]; w fp32 [M, 6] (renormalized routing weights, no 1.5); raw = b2_common.load_experts_raw
    (w13 = [w1 (gate); w3 (up)] rows).  xq: optional already-dequantized MXFP8 x (e.g. flashinfer's own mxfp8_quantize)."""
    M = x.shape[0]
    inter = raw['inter']
    xd = quant_mxfp8(x)[0] if xq is None else xq.float()
    wb = w.to(torch.bfloat16).float()   # PackTopkIds: routing weights travel as bf16
    out = torch.zeros((M, x.shape[1]), dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        w13 = dequant_mxfp4(raw['w13'][e], raw['s13'][e])   # [2I, DIM]
        y = xd[tok] @ w13.t()
        act = swiglu_clamp(y[:, :inter], y[:, inter:])
        if act_bf16:
            act = act.bfloat16().float()
        h = quant_mxfp8(act, rule='floor')[0]
        o = h @ dequant_mxfp4(raw['w2'][e], raw['s2'][e]).t()   # [n, DIM]
        out.index_add_(0, tok, o * wb[tok, slot].unsqueeze(1))
    return out.bfloat16()


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30))


def tok_rel_max(a, b):
    a, b = a.float(), b.float()
    return float(((a - b).square().sum(-1).sqrt() / b.square().sum(-1).sqrt().clamp_min(1e-30)).max())
