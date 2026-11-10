"""Production-style TP8 baseline arm ("trt"): flashinfer trtllm-gen MXFP4 x MXFP8 routed MoE on this rank's TP slice (same bf16 weights,
same MXFP4 quantization as ours, prepared with flashinfer's own row reorder / shuffle / SF interleave as SGLang does), x quantized inside
the timed graph by flashinfer mxfp8_quantize, then a [M, 5120] bf16 all-reduce: flashinfer TRT-LLM one-shot / two-shot IPC all-reduce
(allreduce_fusion, pattern kAllReduce, backend 'trtllm') or NCCL (arm "trtn")."""
import torch

import v41host as H


def _imports():
    import flashinfer
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache
    return flashinfer, trtllm_fp4_block_scale_routed_moe, _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache


def prep_weights(raw, ipad=None, up_first=True, sf_elts=16, etm=128):
    """raw MXFP4 slices (v41host.quantize_slices: w13 [E][576][2560] gate rows 0..287 | up rows 288..575, s13 [E][576][160] E8M0;
    w2 [E][5120][144], s2 [E][5120][9]) -> trtllm-gen tensors.  ipad > 288 zero-pads the intermediate dim (zero weights, scale 2^0)."""
    fi, _, w31_perm, w2_perm = _imports()
    I = H.INTER
    ipad = ipad or I
    E, dev = raw['w13'].shape[0], raw['w13'].device
    g, u = raw['w13'][:, :I], raw['w13'][:, I:]
    gs, us = raw['s13'][:, :I], raw['s13'][:, I:]

    def padr(t, fill):
        if ipad == I:
            return t
        p = torch.full((E, ipad - I, t.shape[2]), fill, dtype=t.dtype, device=dev)
        return torch.cat([t, p], 1)
    first, second = ((u, us), (g, gs)) if up_first else ((g, gs), (u, us))
    w13 = torch.cat([padr(first[0], 0), padr(second[0], 0)], 1).contiguous()          # [E][2 ipad][2560]
    s13 = torch.cat([padr(first[1], 127), padr(second[1], 127)], 1).contiguous()      # [E][2 ipad][160]
    w2, s2 = raw['w2'], raw['s2']
    if ipad != I:
        w2 = torch.cat([w2, torch.zeros(E, H.DIM, (ipad - I) // 2, dtype=torch.uint8, device=dev)], 2)
        s2 = torch.cat([s2, torch.full((E, H.DIM, (ipad - I) // 32), 127, dtype=torch.uint8, device=dev)], 2)
    w2, s2 = w2.contiguous(), s2.contiguous()
    cache = {}
    o13, os13, o2, os2 = [], [], [], []
    for e in range(E):
        p = w31_perm(cache, w13[e], etm)
        o13.append(w13[e][p.to(dev)].contiguous())
        p = w31_perm(cache, s13[e], etm, num_elts_per_sf=sf_elts)
        os13.append(fi.block_scale_interleave(s13[e][p.to(dev)].contiguous()))
        p = w2_perm(cache, w2[e], etm)
        o2.append(w2[e][p.to(dev)].contiguous())
        p = w2_perm(cache, s2[e], etm, num_elts_per_sf=sf_elts)
        os2.append(fi.block_scale_interleave(s2[e][p.to(dev)].contiguous()))
    f8 = torch.float8_e4m3fn
    return dict(w13=torch.stack(o13), s13=torch.stack(os13).view(f8).reshape(E, 2 * ipad, -1),
                w2=torch.stack(o2), s2=torch.stack(os2).view(f8).reshape(E, H.DIM, -1), ipad=ipad)


class TrtMoE:
    def __init__(self, W, dev, beta=0.0, method=None, pdl=True, valid_i=None):
        fi, op, _, _ = _imports()
        self.fi, self.op, self.W = fi, op, W
        E = H.NEXP
        f32 = dict(dtype=torch.float32, device=dev)
        self.alpha = torch.ones(E, **f32)
        self.beta = None if beta is None else torch.full((E,), beta, **f32)
        self.clamp = torch.full((E,), H.CLAMP, **f32)
        self.ones = torch.ones(E, **f32)
        from flashinfer.tllm_enums import RoutingMethodType
        self.method = int(RoutingMethodType.TopK if method is None else method)
        self.pdl = pdl
        self.kw = {} if not valid_i else dict(valid_intermediate_size=valid_i)

    def __call__(self, x, ids, rw, out):
        """out [M, DIM] bf16 = this rank's routed sum (x bf16 [M, DIM], ids int32 [M][6], rw fp32 [M][6])"""
        M = x.shape[0]
        xq, xsf = self.fi.mxfp8_quantize(x, False)
        W = self.W
        self.op((ids, rw), None, xq, xsf.view(torch.float8_e4m3fn).reshape(M, -1), W['w13'], W['s13'], None, self.alpha, self.beta, self.clamp,
                W['w2'], W['s2'], None, self.ones, self.ones, self.ones, H.NEXP, H.TOPK, None, None, W['ipad'], 0, H.NEXP, None,
                routing_method_type=self.method, do_finalize=True, enable_pdl=self.pdl, output=out, tune_max_num_tokens=512, **self.kw)
        return out


def autotune_ctx():
    try:
        from flashinfer.autotuner import autotune
    except ImportError:
        from flashinfer import autotune
    return autotune(True)


class TrtAR:
    """flashinfer TRT-LLM IPC all-reduce (one-shot / two-shot chosen by flashinfer), bf16 [M, DIM]"""

    def __init__(self, rank, group, max_tokens=512, pdl=True):
        import flashinfer.comm as C
        self.C = C
        self.ws = C.create_allreduce_fusion_workspace(backend='trtllm', world_size=8, rank=rank, max_token_num=max_tokens, hidden_dim=H.DIM,
                                                      dtype=torch.bfloat16, group=group)
        self.pdl = pdl

    def __call__(self, inp, out):
        self.C.allreduce_fusion(inp, self.ws, pattern=int(self.C.AllReduceFusionPattern.kAllReduce), launch_with_pdl=self.pdl, output=out)
        return out
