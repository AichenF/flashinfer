"""B200 (sm_100a) TP shard loading for DeepSeek-V4-Flash layer tests: the rank's routed experts as SGLang's
Mxfp4FlashinferTrtllmMoEMethod holds them (`--moe-runner-backend flashinfer_mxfp4`, the B200 TP4 recipe), the shared expert as
its Fp8LinearMethod holds it on SM100 (DeepGEMM, UE8M0 requant), the router, and the production callables of one MoE layer."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

DIM, NEXP, TOPK, INTER_ALL = 4096, 256, 6, 2048


def weight_map(model):
    return json.loads((Path(model) / 'model.safetensors.index.json').read_text())['weight_map']


def load_experts_raw(model, wm, index, ndev, rank, dev):
    """The rank's expert shard as FusedMoE holds it after loading: w13 = [w1 rows; w3 rows] int8 [E, 2I, DIM/2], scales fp32
    (E8M0 values) [E, 2I, DIM/32]; w2 = the rank's K columns int8 [E, DIM, I/2], scales [E, DIM, I/32]."""
    from safetensors import safe_open
    model = Path(model)
    inter = INTER_ALL // ndev
    r0, r1 = inter * rank, inter * (rank + 1)
    files = {}
    for e in range(NEXP):
        for n in ['w1', 'w3', 'w2']:
            for kind in ['weight', 'scale']:
                k = f'layers.{index}.ffn.experts.{e}.{n}.{kind}'
                files.setdefault(wm[k], []).append(k)
    t = {}
    for fn, keys in files.items():
        with safe_open(str(model / fn), framework='pt', device='cpu') as f:
            for k in keys:
                s = f.get_slice(k)
                if '.w2.' in k:   # columns: packed fp4 bytes inter/2 per rank, e8m0 scales inter/32
                    c = (inter // 2) if k.endswith('weight') else (inter // 32)
                    t[k] = s[:, c * rank:c * (rank + 1)]
                else:
                    t[k] = s[r0:r1]
    p = f'layers.{index}.ffn.experts'
    w13 = torch.stack([torch.cat([t[f'{p}.{e}.w1.weight'], t[f'{p}.{e}.w3.weight']]) for e in range(NEXP)])
    s13 = torch.stack([torch.cat([t[f'{p}.{e}.w1.scale'], t[f'{p}.{e}.w3.scale']]) for e in range(NEXP)])
    w2 = torch.stack([t[f'{p}.{e}.w2.weight'] for e in range(NEXP)])
    s2 = torch.stack([t[f'{p}.{e}.w2.scale'] for e in range(NEXP)])
    assert w13.dtype in (torch.int8, torch.uint8) and w13.shape == (NEXP, 2 * inter, DIM // 2), (w13.dtype, w13.shape)
    assert w2.shape == (NEXP, DIM, inter // 2) and s13.shape == (NEXP, 2 * inter, DIM // 32) and s2.shape == (NEXP, DIM, inter // 32)
    return dict(w13=w13.view(torch.int8).contiguous().to(dev), s13=s13.float().contiguous().to(dev),
                w2=w2.view(torch.int8).contiguous().to(dev), s2=s2.float().contiguous().to(dev), inter=inter)


class _NoFp8:   # the FP8 base method's FP4-expert step is a dtype view (fp8.py, is_fp4_expert): done by load_experts_raw already
    def process_weights_after_loading(self, layer):
        pass


def sglang_trtllm_layer(raw, dev):
    """Run SGLang's own Mxfp4FlashinferTrtllmMoEMethod.process_weights_after_loading on the raw shard (w3/w1 reorder, gated
    interleave + shuffle_matrix_a rows, block_scale_interleave SF) -> the layer the trtllm-gen routed MoE reads."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import Mxfp4FlashinferTrtllmMoEMethod
    P = torch.nn.Parameter
    layer = torch.nn.Module()
    layer.num_experts = layer.num_local_experts = NEXP
    layer.moe_ep_rank = 0
    layer.intermediate_size_per_partition, layer.hidden_size = raw['inter'], DIM
    layer.register_parameter('w13_weight', P(raw['w13'].clone(), requires_grad=False))
    layer.register_parameter('w2_weight', P(raw['w2'].clone(), requires_grad=False))
    layer.register_parameter('w13_weight_scale_inv', P(raw['s13'].clone(), requires_grad=False))
    layer.register_parameter('w2_weight_scale_inv', P(raw['s2'].clone(), requires_grad=False))
    m = object.__new__(Mxfp4FlashinferTrtllmMoEMethod)   # (__init__ only reads server args for the precision: "default")
    m._fp8, m.prefix, m.flashinfer_mxfp4_moe_precision = _NoFp8(), f'b2.experts', 'default'
    m.process_weights_after_loading(layer)
    m.create_moe_runner(layer, SimpleNamespace(swiglu_limit=10.0))
    layer.quant_method = m
    return layer


def load_shared(model, wm, index, ndev, rank, dev):
    """The rank's shared-expert slices (gate_up = [w1; w3] rows, down = w2 columns) with SGLang's SM100 DeepGEMM processing
    (requant_weight_ue8m0: block scales re-quantized to UE8M0 and packed MN-major)."""
    from safetensors import safe_open
    from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0
    model = Path(model)
    inter = INTER_ALL // ndev
    nb = inter // 128

    def ck(key):
        with safe_open(str(model / wm[key]), framework='pt', device='cpu') as f:
            return f.get_tensor(key).to(dev)
    p = f'layers.{index}.ffn.shared_experts'
    w1, w3, w2 = (ck(f'{p}.{n}.weight') for n in ['w1', 'w3', 'w2'])
    s1, s3, s2 = (ck(f'{p}.{n}.scale').float() for n in ['w1', 'w3', 'w2'])
    gu_w = torch.cat([w1[inter * rank:inter * (rank + 1)], w3[inter * rank:inter * (rank + 1)]]).contiguous()
    gu_s = torch.cat([s1[nb * rank:nb * (rank + 1)], s3[nb * rank:nb * (rank + 1)]]).contiguous()
    dn_w = w2[:, inter * rank:inter * (rank + 1)].contiguous()
    dn_s = s2[:, nb * rank:nb * (rank + 1)].contiguous()
    raw = dict(gu_w=gu_w, gu_s=gu_s, dn_w=dn_w, dn_s=dn_s)   # checkpoint numerics (fp32 block scales), for the reference
    gw, gs = requant_weight_ue8m0(gu_w, gu_s, [128, 128])
    dw, ds = requant_weight_ue8m0(dn_w, dn_s, [128, 128])
    # the same requantization with the UE8M0 block scales left as floats [row blocks, k blocks] (the fused kernel's SFA images)
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant, quant_weight_ue8m0
    gw2, gs_f = quant_weight_ue8m0(block_quant_dequant(gu_w, gu_s, [128, 128], torch.bfloat16), [128, 128])
    dw2, ds_f = quant_weight_ue8m0(block_quant_dequant(dn_w, dn_s, [128, 128], torch.bfloat16), [128, 128])
    assert torch.equal(gw2.view(torch.uint8), gw.view(torch.uint8)) and torch.equal(dw2.view(torch.uint8), dw.view(torch.uint8)), 'requant mismatch'
    return dict(raw=raw, gu_w=gw, gu_s=gs, dn_w=dw, dn_s=ds, gs_f=gs_f, ds_f=ds_f, inter=inter)


def load_router(model, wm, index, dev):
    from safetensors import safe_open
    model = Path(model)

    def ck(key):
        with safe_open(str(model / wm[key]), framework='pt', device='cpu') as f:
            return f.get_tensor(key).to(dev)
    return ck(f'layers.{index}.ffn.gate.weight').bfloat16().contiguous(), ck(f'layers.{index}.ffn.gate.bias').float().contiguous()


def capture_pool(cap, ms, dev, layer_key='layer-20-module-20', off=0):
    rows = []
    for label in ['c8', 'c4', 'c1']:
        batch = torch.load(Path(cap) / (label + '.pt'), map_location='cpu', weights_only=False)
        rows += [v['x'] for (key, _), v in batch.items() if key == layer_key]
    pool = torch.cat(rows).to(dev).bfloat16().contiguous()
    while pool.shape[0] < max(ms) + off:
        pool = torch.cat([pool, (pool * 1.01).bfloat16()])
    return {m: pool[off:off + m].contiguous() for m in ms}


class Prod:
    """One MoE layer of DeepseekV2MoE.forward_normal_dual_stream on SM100 with --moe-runner-backend flashinfer_mxfp4:
    router GEMM (dsv3_router_gemm M <= 4, else cuBLAS fp32 out) -> moe_fused_gate (sqrtsoftplus, renorm) -> PackTopkIds ->
    mxfp8_quantize (cute-dsl) -> trtllm_fp4_block_scale_routed_moe (do_finalize) on the main stream; the shared expert
    (DeepGEMM FP8, UE8M0) on the alt stream; shared.add_(routed, alpha=1.5); custom all-reduce."""

    def __init__(self, layer, shared, router, bias, alt, ar):
        self.layer, self.sh, self.router, self.bias, self.alt, self.ar = layer, shared, router, bias, alt, ar

    def logits(self, x):
        from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
        from sglang.jit_kernel.dsv4.gemm import linear_bf16_fp32
        return dsv3_router_gemm(x, self.router, out_dtype=torch.float32) if x.shape[0] <= 4 else linear_bf16_fp32(x, self.router)

    def topk(self, lg):
        from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
        w, i = moe_fused_gate(lg, self.bias, topk=TOPK, scoring_func='sqrtsoftplus', num_fused_shared_experts=0, renormalize=True,
                              routed_scaling_factor=1.5, apply_routed_scaling_factor_on_output=False)
        return w, i

    def routed(self, x, w, i):
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
        from sglang.srt.layers.moe.utils import RoutingMethodType
        from sglang.srt.utils.common import next_power_of_2
        L = self.layer
        packed = PackTopkIds.execute(i.to(torch.int32).contiguous(), w.float().contiguous())
        xq, xs = mxfp8_quantize(x, False, alignment=DIM, backend='cute-dsl')
        xs = xs.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
        out = torch.empty((x.shape[0], DIM), dtype=torch.bfloat16, device=x.device)
        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed, routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
            gemm1_weights=L.w13_weight, gemm1_weights_scale=L.w13_weight_scale_inv, gemm1_bias=None, gemm1_alpha=None,
            gemm1_beta=None, gemm1_clamp_limit=L.quant_method._gemm1_clamp_limit_tensor, gemm2_weights=L.w2_weight,
            gemm2_weights_scale=L.w2_weight_scale_inv, gemm2_bias=None, output1_scale_scalar=L.output1_scale_scalar,
            output1_scale_gate_scalar=L.output1_scale_gate_scalar, output2_scale_scalar=L.output2_scale_scalar,
            num_experts=NEXP, top_k=TOPK, n_group=1, topk_group=1, intermediate_size=self.sh['inter'],
            local_expert_offset=0, local_num_experts=NEXP, routed_scaling_factor=1.0,
            routing_method_type=int(RoutingMethodType.TopK), do_finalize=True,
            tune_max_num_tokens=next_power_of_2(x.shape[0]), output=out)[0]

    def shared(self, x):
        from sglang.jit_kernel.dsv4.moe import silu_and_mul_clamp
        from sglang.srt.layers.quantization.fp8_utils import deepgemm_w8a8_block_fp8_linear_with_fallback as lin
        S = self.sh
        gu = lin(x, S['gu_w'], [128, 128], S['gu_s'])
        act = torch.empty((x.shape[0], S['inter']), dtype=torch.bfloat16, device=x.device)
        silu_and_mul_clamp(gu, act, 10.0)
        return lin(act, S['dn_w'], [128, 128], S['dn_s'])

    def __call__(self, x, with_ar=True):   # forward_normal_dual_stream
        cur = torch.cuda.current_stream()
        self.alt.wait_stream(cur)
        lg = self.logits(x)
        w, i = self.topk(lg)
        r = self.routed(x, w, i)
        with torch.cuda.stream(self.alt):
            s = self.shared(x)
        cur.wait_stream(self.alt)
        s.add_(r, alpha=1.5)
        return self.ar.custom_all_reduce(s) if with_ar else s
