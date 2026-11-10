"""flashinfer trtllm-gen TP baseline arm (generalized from tp8_v41_b300/trtarm.py): this rank's TP slice as SGLang's
Mxfp4FlashinferTrtllmMoEMethod holds it (w3/w1 reorder = up rows first, flashinfer row permutes, block_scale_interleave SF; the same
tensors feed our v77 kernel), x quantized in the timed graph by flashinfer mxfp8_quantize, trtllm_fp4_block_scale_routed_moe
(MXFP4 x MXFP8, do_finalize), then the [M, DIM] bf16 all-reduce: flashinfer TRT-LLM IPC one-/two-shot (allreduce_fusion kAllReduce)."""
import torch


def _imports():
    import flashinfer
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache
    return flashinfer, trtllm_fp4_block_scale_routed_moe, _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache


def prep_weights(sh, raw, ipad=None, sf_elts=16, etm=128):
    """raw MXFP4 slices (sbcommon.quantize_slices: w13 gate rows | up rows) -> trtllm-gen tensors (SGLang processing).  ipad > I_r
    zero-pads the intermediate dim (zero weights, scale 2^0)."""
    fi, _, w31_perm, w2_perm = _imports()
    I = sh.INTER
    ipad = ipad or I
    E, dev = raw['w13'].shape[0], raw['w13'].device
    g, u = raw['w13'][:, :I], raw['w13'][:, I:]
    gs, us = raw['s13'][:, :I], raw['s13'][:, I:]

    def padr(t, fill):
        if ipad == I:
            return t
        p = torch.full((E, ipad - I, t.shape[2]), fill, dtype=t.dtype, device=dev)
        return torch.cat([t, p], 1)
    w13 = torch.cat([padr(u, 0), padr(g, 0)], 1).contiguous()          # [E][2 ipad][DIM/2]  (w3 = up first, as SGLang's reorder)
    s13 = torch.cat([padr(us, 127), padr(gs, 127)], 1).contiguous()
    w2, s2 = raw['w2'], raw['s2']
    if ipad != I:
        w2 = torch.cat([w2, torch.zeros(E, sh.DIM, (ipad - I) // 2, dtype=torch.uint8, device=dev)], 2)
        s2 = torch.cat([s2, torch.full((E, sh.DIM, (ipad - I) // 32), 127, dtype=torch.uint8, device=dev)], 2)
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
                w2=torch.stack(o2), s2=torch.stack(os2).view(f8).reshape(E, sh.DIM, -1), ipad=ipad)


class TrtMoE:
    def __init__(self, sh, W, dev, pdl=True):
        fi, op, _, _ = _imports()
        self.sh, self.fi, self.op, self.W = sh, fi, op, W
        E = sh.NEXP
        f32 = dict(dtype=torch.float32, device=dev)
        self.clamp = torch.full((E,), 10.0, **f32)
        self.ones = torch.ones(E, **f32)
        from flashinfer.tllm_enums import RoutingMethodType
        self.method = int(RoutingMethodType.TopK)
        self.pdl = pdl

    def __call__(self, x, ids, rw, out):
        """out [M, DIM] bf16 = this rank's routed sum (x bf16 [M, DIM], ids int32 [M][6], rw fp32 [M][6])"""
        M = x.shape[0]
        sh, W = self.sh, self.W
        xq, xsf = self.fi.mxfp8_quantize(x, False)
        # (as SGLang's routed path: gemm1 alpha / beta None, clamp 10, output scales 1)
        self.op((ids, rw), None, xq, xsf.view(torch.float8_e4m3fn).reshape(M, -1), W['w13'], W['s13'], None, None, None, self.clamp,
                W['w2'], W['s2'], None, self.ones, self.ones, self.ones, sh.NEXP, sh.TOPK, None, None, W['ipad'], 0, sh.NEXP, None,
                routing_method_type=self.method, do_finalize=True, enable_pdl=self.pdl, output=out, tune_max_num_tokens=512)
        return out


def autotune_ctx():
    try:
        from flashinfer.autotuner import autotune
    except ImportError:
        from flashinfer import autotune
    return autotune(True)


class TrtAR:
    """flashinfer TRT-LLM IPC all-reduce (one-shot / two-shot chosen by flashinfer), bf16 [M, DIM]"""

    def __init__(self, sh, rank, group, max_tokens=512, pdl=True):
        import flashinfer.comm as C
        self.C = C
        self.ws = C.create_allreduce_fusion_workspace(backend='trtllm', world_size=sh.ndev, rank=rank, max_token_num=max_tokens,
                                                      hidden_dim=sh.DIM, dtype=torch.bfloat16, group=group)
        self.pdl = pdl

    def __call__(self, inp, out):
        self.C.allreduce_fusion(inp, self.ws, pattern=int(self.C.AllReduceFusionPattern.kAllReduce), launch_with_pdl=self.pdl, output=out)
        return out
