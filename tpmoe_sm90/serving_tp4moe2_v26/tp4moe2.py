"""Host side of tp4moe2.cu: extension loader, per-rank workspace (+ LL all-reduce buffers), and the bench adapter.

Inputs at the bench3/bench4 boundary: Humming-quantized x (e4m3 [M,4096] + per-token fp32 scale), routing ids/weights [M,k],
BF16 shared-expert output [M,4096].  Output: BF16 [M,4096], TP4-reduced (ndev=4) or rank-local (ndev=1:
BF16(BF16(routed * factor) + shared)).
"""
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def ext():
    global _ext
    if _ext is None:
        flags = os.environ.get('TP2_FLAGS', '').split()
        src = os.environ.get('TP2_SRC', 'tp4moe2.cu')   # A/B against a snapshot (same host code)
        tag = '' if src == 'tp4moe2.cu' else '_' + ''.join(c if c.isalnum() else '_' for c in src[:-3])
        name = 'tp4moe2_shared' + tag + ('_' + ''.join(c if c.isalnum() else '_' for c in ''.join(flags)) if flags else '')
        _ext = load(name=name, sources=[os.path.join(HERE, src)],
                    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-I{HERE}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'] + flags,
                    verbose=False)
    return _ext


def humming_layer_weights(layer):
    return dict(w1=layer.w13_weight.data, s1=layer.w13_weight_scale.data.view(torch.uint8), f1=layer.w13_global_scale.data.reshape(-1).float(),
                w2=layer.w2_weight.data, s2=layer.w2_weight_scale.data.view(torch.uint8), f2=layer.w2_global_scale.data.reshape(-1).float())


class RankState:
    """Per-rank workspace.  Counters / flags are epoch-tagged or parity-reset by the kernel itself (no host resets)."""

    def __init__(self, device, alloc_comm=True):
        E = ext()
        self.device = device
        grid = torch.cuda.get_device_properties(device).multi_processor_count
        z = dict(device=device)
        self.act = torch.zeros(E.MAXE * 512, dtype=torch.bfloat16, **z)
        self.amax = torch.zeros(E.MAXE * 4, dtype=torch.float32, **z)
        self.part = torch.zeros(2 * grid * E.NCONS * 32, dtype=torch.float32, **z)
        self.act_flags = torch.zeros(E.MAXP * 4, dtype=torch.int32, **z)
        self.part_flags = torch.zeros(max(2 * grid, 256), dtype=torch.int32, **z)
        self.rt_cnt = torch.zeros(2 * E.NRT, dtype=torch.int32, **z)
        self.work = torch.zeros(4, dtype=torch.int32, **z)
        self.outp = torch.zeros(E.MAXM * E.TOPK_MAX * 4096, dtype=torch.bfloat16, **z)
        self.hq = torch.zeros(E.MAXP * 4 * E.NMAX * 128, dtype=torch.uint8, **z)
        self.hs = torch.zeros(E.MAXP * E.NMAX, dtype=torch.float32, **z)
        self.h_flags = torch.zeros(E.MAXP, dtype=torch.int32, **z)
        self.pan_cnt = torch.zeros(2 * E.MAXP, dtype=torch.int32, **z)
        self.xg = torch.zeros((E.MAXE + 8 * E.MAXP) * 4096, dtype=torch.uint8, **z)
        self.xg_cnt = torch.zeros(2 * E.MAXP, dtype=torch.int32, **z)
        self.fc2_next = torch.zeros(4, dtype=torch.int32, **z)   # [FC2 unit queue x 2 parities, T1 item queue x 2 parities]
        # fused input quantization (run(..., xb=BF16 x)): quantized x / per-token scale workspace + per-token ready flags
        self.xqw = torch.zeros(max(E.MAXM, grid * 16) * 4096, dtype=torch.uint8, **z)   # (dense: per-CTA 16-row local quantization slots)
        self.xsw = torch.zeros(E.MAXM, dtype=torch.float32, **z)
        self.xflag = torch.zeros(E.MAXM, dtype=torch.int32, **z)
        self.seq = 0
        # fused routing (run(..., gbias=..)): ids / weights outputs, [logits fp32 MAXM x 256 | token flags MAXM | barrier counters]
        self.gids = torch.zeros(E.MAXM * 8, dtype=torch.int32, **z)
        self.grw = torch.zeros(E.MAXM * 8, dtype=torch.float32, **z)
        self.gws = torch.zeros(E.MAXM * 256 + E.MAXM + 8, dtype=torch.int32, **z)
        # fused shared expert (run(..., sh=...)): group-quantized x [MAXM,4096] + scales [32,MAXM], FC2 input group scales, output
        self.xqg = torch.zeros(E.MAXM * 4096, dtype=torch.uint8, **z)
        self.xsT = torch.zeros(32 * E.MAXM, dtype=torch.float32, **z)
        self.hs4 = torch.zeros(E.MAXP * 4 * E.NMAX, dtype=torch.float32, **z)
        self.shout = torch.zeros(E.MAXM * 4096, dtype=torch.bfloat16, **z)
        self.rs_ptrs, self.ag_ptrs = [], []
        if alloc_comm:
            self.rs_buf = torch.zeros(self.RS_BYTES() // 4, dtype=torch.int32, **z)
            self.ag_buf = torch.zeros(self.AG_BYTES() // 4, dtype=torch.int32, **z)

    @staticmethod
    def RS_BYTES():
        E = ext()
        return 16 * E.MAXM * E.NMSG * 16

    @staticmethod
    def AG_BYTES():
        E = ext()
        return E.NRT * E.MAXM * E.NMSG * 16

    def link_ptrs(self, rs_ptrs, ag_ptrs):
        self.rs_ptrs = [int(p) for p in rs_ptrs]
        self.ag_ptrs = [int(p) for p in ag_ptrs]

    @staticmethod
    def link(states):
        rs = [s.rs_buf.data_ptr() for s in states]
        ag = [s.ag_buf.data_ptr() for s in states]
        for s in states:
            s.link_ptrs(rs, ag)


_EMPTY = {}


def _empty(device, dtype):
    key = (str(device), dtype)
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, dtype=dtype, device=device)
    return _EMPTY[key]


def shared_weights(mlp):
    """production DeepseekV2MLP (fp8 block-quant) -> the fused shared expert's weights (used in place)"""
    g, d = mlp.gate_up_proj, mlp.down_proj
    return dict(w1=g.weight.data.view(torch.uint8), s1=g.weight_scale_inv.data.float().contiguous(),
                w2=d.weight.data.view(torch.uint8), s2=d.weight_scale_inv.data.float().contiguous())


def run(state, w, xq, xs, ids, rw, shared, factor, out, my_rank, ndev, dbg=None, tl=None, xb=None, glog=None, gbias=None, gw=None, topk=6, sh=None):
    """xb: BF16 x [M,4096] -> the kernel quantizes it itself (Humming per-token e4m3, bit for bit); xq / xs are then ignored.
    gbias (fp32 [256]): fused routing (moe_fused_gate sqrtsoftplus, renormalize, bit for bit) of the logits glog (fp32 [M,256]) or, with
    gw (router weight bf16 [256,4096], M <= 16), of the in-kernel router GEMM (dsv3_router_gemm, bit for bit); ids / rw are then the
    workspace outputs (state.gids / state.grw [:M*topk])."""
    if xb is not None:
        state.seq += 1
        xq, xs = state.xqw, state.xsw
        xf, seq = state.xflag, state.seq
    else:
        xf, seq = _empty(xq.device, torch.int32), 0
    dev = out.device
    if gbias is not None:
        m = out.shape[0]
        ids, rw = state.gids[:m * topk].view(m, topk), state.grw[:m * topk].view(m, topk)
    ext().run(xq.view(torch.uint8), xs, w['w1'], w['s1'], w['f1'], w['w2'], w['s2'], w['f2'], ids, rw,
              shared if shared is not None else _empty(xq.device, torch.bfloat16), float(factor),
              state.act, state.amax, state.part, state.act_flags, state.part_flags, state.rt_cnt, state.work, state.outp,
              state.hq, state.hs, state.h_flags, state.pan_cnt, state.xg, state.xg_cnt, state.fc2_next,
              state.rs_ptrs if ndev > 1 else [], state.ag_ptrs if ndev > 1 else [], out, my_rank, ndev,
              dbg if dbg is not None else _empty(xq.device, torch.int64), tl if tl is not None else _empty(xq.device, torch.int64),
              xb if xb is not None else _empty(dev, torch.bfloat16), xf, seq,
              glog if glog is not None else _empty(dev, torch.float32), gbias if gbias is not None else _empty(dev, torch.float32),
              gw if gw is not None else _empty(dev, torch.bfloat16), state.gws,
              *((sh['w1'], sh['s1'], sh['w2'], sh['s2'], state.xqg, state.xsT, state.hs4, state.shout) if sh is not None else
                (_empty(dev, torch.uint8), _empty(dev, torch.float32), _empty(dev, torch.uint8), _empty(dev, torch.float32),
                 state.xqg, state.xsT, state.hs4, state.shout)))
    return out


class TP4Op:
    """bench adapter: tp4() = the TP4 entry (LL all-reduce over the 4 ranks), local() = rank-local BF16(routed * factor)."""

    def __init__(self, layer, dev, rank, linked, ipc_link=None, factor=1.5, state=None):
        self.w = humming_layer_weights(layer)
        self.dev, self.rank, self.factor = dev, rank, factor
        if state is not None:
            self.st = state
        else:
            self.st = RankState(dev, alloc_comm=not linked)
            if linked:
                ipc_link(self.st, RankState.RS_BYTES(), RankState.AG_BYTES())
        self.ext = ext()

    def tp4(self, xq, xs, ids, prob, shared):
        out = torch.empty((xq.shape[0], 4096), dtype=torch.bfloat16, device=self.dev)
        return run(self.st, self.w, xq, xs, ids, prob, shared, self.factor, out, self.rank, 4)

    def tp4x(self, x, ids, prob, shared, sh=None):   # fused input quantization (BF16 x); sh: fused shared expert (then shared is None)
        out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=self.dev)
        return run(self.st, self.w, None, None, ids, prob, shared, self.factor, out, self.rank, 4, xb=x, sh=sh)

    def tp4g(self, x, router, bias, shared, logits=None, sh=None):   # fused routing + input quantization: router GEMM in-kernel if logits is None
        out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=self.dev)
        return run(self.st, self.w, None, None, None, None, shared, self.factor, out, self.rank, 4, xb=x,
                   glog=logits, gbias=bias, gw=router if logits is None else None, sh=sh)

    def router16(self, x, router):   # dsv3_router_gemm bit for bit (M <= 16), x shared in smem by 4 experts per block
        out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=self.dev)
        ext().router16(x, router, out)
        return out

    def routerx(self, x, router):   # dsv3_router_gemm bit for bit (M <= 16): x lanes reused for EB experts per block, early PDL trigger
        out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=self.dev)
        ext().routerx(x, router, out, 1 if x.shape[0] <= 8 else 2)
        return out

    def localg(self, x, router, bias, logits=None, shared=None, sh=None):
        out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=self.dev)
        return run(self.st, self.w, None, None, None, None, shared, self.factor, out, 0, 1, xb=x,
                   glog=logits, gbias=bias, gw=router if logits is None else None, sh=sh)

    def localx(self, x, ids, prob, shared=None, sh=None):   # rank-local: BF16(BF16(routed * factor) + shared), shared fused if sh
        out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=self.dev)
        return run(self.st, self.w, None, None, ids, prob, shared, self.factor, out, 0, 1, xb=x, sh=sh)

    def local(self, xq, xs, ids, prob):
        out = torch.empty((xq.shape[0], 4096), dtype=torch.bfloat16, device=self.dev)
        return run(self.st, self.w, xq, xs, ids, prob, None, self.factor, out, 0, 1)
