"""Serve the frozen tp4moe2 kernels in the DeepSeek-V4 TP4 decode worker (FAC_FMOE_MODE=tp4moe2): two builds sharing one per-rank
workspace -- 'sh' (v24: the shared expert fused, one N32 instantiation) for M <= FAC_TP4_SH_FUSED_MAXM and 'ns' (v23 with the shared
code compiled out: inlined, the shared expert's consumer code slowed the routed loops by 2..9 us) for larger M.

Scope: DeepseekV2MoE.forward_normal_dual_stream / forward_normal of every MoE layer (target and DSPARK draft) whose experts use
Humming MXFP4 (W4A8, fused E8M0, per-token e4m3 input).  One call replaces production
  route (dsv3_router_gemm | linear_bf16_fp32 + noaux top-k, or hash top-k) -> Humming experts (x1.5) -> + shared -> TP all-reduce
with
  route logits (bit-identical router GEMM) -> tp4moe2 v20 (fused gate | given top-k, input quant, FC1/FC2, x1.5, + shared, LL all-reduce).
The shared expert: M <= FAC_TP4_SH_FUSED_MAXM (32) or M >= FAC_TP4_SH_FUSED_FROMM (H100 TP4: <= 21 or >= 33) inside the kernel (production FP8 weights in place, production numerics -- where
the M <= 16 router GEMM is too short to hide it); larger M the production module on the alt stream, overlapped with the router GEMM
(cuBLAS, ~7 us) and joined before the kernel (bench6: fused +6..19 us there, overlapped ~0..8 us).  Everything else -- unsupported shapes / flags,
M > MAXM, EPLB, reduce-scatter combine -- falls back to the original methods.

TP8 (tensor-parallel world size 8, per-rank intermediate 256): both builds compiled with -DTP2_NDEV=8 (per-rank shared expert
gate_up [512, 4096], down [4096, 256]).

FAC_TP4_SHADOW=N (default 1): for eager (non-capturing) calls, the first N calls per (layer, M) also run the original path and
log the difference (CUDA-graph warmup runs every captured M eagerly twice before capture, so startup covers all graph sizes).
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

OUT = Path(__file__).resolve().parent
assert os.environ.get('FAC_FMOE_MODE') == 'tp4moe2'
FROZEN = json.loads((OUT / 'sources.json').read_text())
for _name, _digest in FROZEN['sha256'].items():
    assert hashlib.sha256((OUT / _name).read_bytes()).hexdigest() == _digest, _name
sys.path.insert(0, str(OUT))
import importlib.util   # noqa: E402


def _module(name):   # two instances of the host wrapper, one per extension (same workspace layout)
    spec = importlib.util.spec_from_file_location(name, OUT / 'tp4moe2.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _module('tp4moe2_srv_ns')
TS = _module('tp4moe2_srv_sh')

from sglang.srt.layers.quantization.mxfp4_humming_moe import Mxfp4HummingMoEMethod   # noqa: E402
import importlib   # noqa: E402
D2 = importlib.import_module('sglang.srt.models.deepseek_v2')   # (may be called right after its exec, before the package attribute is set)

ORIG_PROCESS = Mxfp4HummingMoEMethod.process_weights_after_loading
ORIG_DUAL = D2.DeepseekV2MoE.forward_normal_dual_stream
ORIG_NORMAL = D2.DeepseekV2MoE.forward_normal
SHADOW = int(os.environ.get('FAC_TP4_SHADOW', '1'))
os.environ.setdefault('TP2_PDL', '1')   # PDL launches (router GEMM -> MoE kernel), as measured in the benches
# fused shared expert up to this M (else the production shared expert on the alt stream, joined before the kernel); default by TP size
# (single-layer A/B: TP4 fused wins at M <= 32 only; TP8 fused wins at every M -- the serialized alt-stream shared costs 13-21 us)
SH_FUSED_MAXM_ENV = os.environ.get('FAC_TP4_SH_FUSED_MAXM')
SH_FUSED_MAXM = int(SH_FUSED_MAXM_ENV) if SH_FUSED_MAXM_ENV else 32
# + fused again from this M up.  H100 TP4 (4-GPU layer A/B, computelab 10-27, 3 runs): fused wins at M <= 21 and M >= 33 (up to
# -9 us), the alt-stream shared expert only at M 22..32 (~3 us) -> defaults 21 / 33 on H100; H20 TP4: 32 / never (unchanged)
SH_FUSED_FROMM_ENV = os.environ.get('FAC_TP4_SH_FUSED_FROMM')
SH_FUSED_FROMM = int(SH_FUSED_FROMM_ENV) if SH_FUSED_FROMM_ENV else 1 << 30
G = dict(state=None, rank=None, keep=[], ready=0, seen=set(), shadow={}, linear=None)


def log(tag, **kw):   # one write per record: the four TP workers share the container's stderr
    sys.stderr.write(tag + ' ' + json.dumps(dict(pid=os.getpid(), **kw)) + '\n')
    sys.stderr.flush()


def ndev():   # tensor-parallel world size of this worker (4 or 8 supported; else 0 -> the hook stays off)
    from sglang.srt.distributed import get_tensor_model_parallel_world_size
    n = get_tensor_model_parallel_world_size()
    return n if n in (4, 8) else 0


def ext():
    if T._ext is None:
        from tp4_ext import load_ext
        n = ndev()
        T._ext = load_ext('ns', n)
        TS._ext = load_ext('sh', n)
        global SH_FUSED_MAXM, SH_FUSED_FROMM
        h100 = n == 4 and 'H100' in torch.cuda.get_device_name()
        if not SH_FUSED_MAXM_ENV:
            SH_FUSED_MAXM = (21 if h100 else 32) if n == 4 else int(T._ext.MAXM)
        if not SH_FUSED_FROMM_ENV and h100:
            SH_FUSED_FROMM = 33
    return T._ext


def link_state(device):
    """One workspace per rank (shared by all layers, as in bench5); LL all-reduce buffers IPC-linked across the TP group."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
    g = get_tp_group()
    E = ext()
    assert g.world_size == E.NDEV, (g.world_size, E.NDEV)
    rank = g.rank_in_group
    st = T.RankState(device, alloc_comm=False)
    lib = CudaRTLibrary()
    tables = []
    for nbytes in [T.RankState.RS_BYTES(), T.RankState.AG_BYTES()]:
        ptr = lib.cudaMalloc(nbytes)
        lib.cudaMemset(ptr, 0, nbytes)
        G['keep'].append(ptr)
        handle = lib.cudaIpcGetMemHandle(ptr)
        raw = [None] * g.world_size
        torch.distributed.all_gather_object(raw, bytes(handle), group=g.cpu_group)
        table = []
        for peer, h in enumerate(raw):
            if peer == rank:
                table.append(ptr.value)
            else:
                p = lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(h))
                G['keep'].append(p)
                table.append(p.value)
        tables.append(table)
    st.link_ptrs(*tables)
    torch.cuda.synchronize(device)
    G['state'], G['rank'] = st, rank
    log('FAC_TP4_LINKED', rank=rank, device=str(device), extension=Path(E.__file__).name,
        extension_sha256=hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest() + (
            '+' + hashlib.sha256(Path(TS._ext.__file__).read_bytes()).hexdigest() if TS._ext is not None else ''),
        maxm=int(E.MAXM), ndev=int(E.NDEV), sh_fused_maxm=SH_FUSED_MAXM, sh_fused_fromm=SH_FUSED_FROMM)


def prepare_weights(self, layer):
    ORIG_PROCESS(self, layer)
    from humming import dtypes
    from humming.config import MmaType
    n = ndev()
    if (not n or layer.hidden_size != 4096 or layer.intermediate_size_per_partition != 2048 // n or layer.num_local_experts != 256):
        return
    for name in ['w13', 'w2']:
        meta = layer.humming_metas[name]
        if not (meta.mma_type == MmaType.WGMMA and meta.use_fused_e8m0_scale
                and meta.b_dtype == dtypes.float4e2m1 and meta.bs_dtype == dtypes.float8e8m0
                and meta.a_dtype == dtypes.float8e4m3 and meta.input_scale_group_size == 0
                and meta.weight_scale_group_size == 32 and not getattr(meta, 'use_packed_k_layout', False)):
            log('FAC_TP4_SKIP_LAYER', reason='humming meta', sublayer=name)
            return
    if G['state'] is None:
        link_state(layer.w13_weight.device)
    layer._fac_tp4_w = T.humming_layer_weights(layer)
    G['ready'] += 1
    log('FAC_TP4_LAYER_READY', count=G['ready'])


def shared_of(self):
    """the production shared expert's FP8 weights, used in place; None (-> original path) if not the expected format"""
    sh = getattr(self, '_fac_sh', 0)
    if sh != 0:
        return sh
    sh = None
    try:
        mlp = self.shared_experts
        g, d = mlp.gate_up_proj, mlp.down_proj
        inter, nqg = int(T._ext.INTER), int(T._ext.NQG)   # per-rank shared intermediate (512 | 256) and its 128-blocks
        ok = (getattr(mlp, 'swiglu_limit', None) == 10.0 and not d.reduce_results
              and g.weight.shape == (2 * inter, 4096) and d.weight.shape == (4096, inter) and g.weight.element_size() == 1 and d.weight.element_size() == 1
              and g.weight.is_contiguous() and d.weight.is_contiguous()
              and g.weight_scale_inv.shape == (2 * nqg, 32) and d.weight_scale_inv.shape == (32, nqg)
              and g.weight_scale_inv.dtype == torch.float32 and d.weight_scale_inv.dtype == torch.float32
              and g.weight_scale_inv.is_contiguous() and d.weight_scale_inv.is_contiguous()
              and not getattr(mlp, '_enable_nvfp4_gemm_swiglu_fusion', False))
        if ok:
            sh = dict(w1=g.weight.data.view(torch.uint8), s1=g.weight_scale_inv.data, w2=d.weight.data.view(torch.uint8), s2=d.weight_scale_inv.data)
        else:
            log('FAC_TP4_SHARED_UNSUPPORTED', layer=self.layer_id, gw=str(tuple(g.weight.shape)), gdt=str(g.weight.dtype),
                gs=str(tuple(g.weight_scale_inv.shape)), gsdt=str(g.weight_scale_inv.dtype), limit=getattr(mlp, 'swiglu_limit', None))
    except AttributeError as e:
        log('FAC_TP4_SHARED_UNSUPPORTED', layer=self.layer_id, error=str(e))
    self._fac_sh = sh
    return sh


def eligible(self, x, skip_shared=False):
    return (G['state'] is not None and getattr(self.experts, '_fac_tp4_w', None) is not None
            and self.tp_size == T._ext.NDEV   # (target and DSPARK draft MoE layers alike: both run Humming in the baseline)
            and x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
            and 1 <= x.shape[0] <= T._ext.MAXM and x.shape[1] == 4096
            and self.num_fused_shared_experts == 0 and hasattr(self, 'shared_experts') and not self._shared_expert_tp1
            and not skip_shared and not self._fuse_shared_experts_inside_sbo
            and not self.experts.should_fuse_routed_scaling_factor_in_topk
            and not D2.get_server_args().enable_eplb
            and not D2.should_skip_post_experts_all_reduce(is_tp_path=True)
            and (TS._ext is None or shared_of(self) is not None))   # (the fused shared expert's weights: TP4 sh build only)


def candidate(self, x, gemm_output_zero_allocator, input_ids_global, use_alt):
    m = x.shape[0]
    st, w = G['state'], self.experts._fac_tp4_w
    fused = TS._ext is not None and (m <= SH_FUSED_MAXM or m >= SH_FUSED_FROMM)
    sh = self._fac_sh if fused else None
    cur = torch.cuda.current_stream()
    shared = None
    if not fused:   # the production shared expert on the alt stream, overlapped with the router GEMM
        if use_alt:
            self.alt_stream.wait_stream(cur)
            with torch.cuda.stream(self.alt_stream):
                shared = self._forward_shared_experts(x, gemm_output_zero_allocator)
        else:
            shared = self._forward_shared_experts(x, gemm_output_zero_allocator)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    if self.is_hash:
        logits = self.gate(x, gemm_output_zero_allocator)
        tk = self.topk(x, logits, expert_location_dispatch_info=None, input_ids=input_ids_global)
        ids = tk.topk_ids if tk.topk_ids.dtype == torch.int32 else tk.topk_ids.int()
        rw = tk.topk_weights if tk.topk_weights.dtype == torch.float32 else tk.topk_weights.float()
        if not fused and use_alt:
            cur.wait_stream(self.alt_stream)
        (TS if fused else T).run(st, w, None, None, ids.contiguous(), rw.contiguous(), shared, self.routed_scaling_factor, out, G['rank'], T._ext.NDEV, xb=x, sh=sh)
    else:
        router = self.gate.weight
        if m <= 16:   # dsv3_router_gemm bit for bit (production uses it for M <= 16 on sm90), early PDL trigger
            logits = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            T._ext.routerx(x, router, logits, 1 if m <= 8 else 2)
        else:
            logits = G['linear'](x, router)
        if not fused and use_alt:
            cur.wait_stream(self.alt_stream)
        (TS if fused else T).run(st, w, None, None, None, None, shared, self.routed_scaling_factor, out, G['rank'], T._ext.NDEV, xb=x,
                                 glog=logits, gbias=self.gate.e_score_correction_bias.data, sh=sh)
    return out


def record(role, layer_id, m):
    key = (role, m)
    if key not in G['seen']:
        G['seen'].add(key)
        log('FAC_TP4_SHAPE', role=role, m=m, layer=layer_id, capturing=torch.cuda.is_current_stream_capturing())


def shadow(self, x, cand, run_original):
    key = (self.layer_id, x.shape[0])
    n = G['shadow'].get(key, 0)
    if n >= SHADOW or torch.cuda.is_current_stream_capturing():
        return
    G['shadow'][key] = n + 1
    ref = run_original()
    a, b = cand.float(), ref.float()
    d = (a - b)
    tok = d.square().mean(1).sqrt() / b.square().mean(1).sqrt().clamp_min(1e-30)   # per token: a routing difference shows here
    log('FAC_TP4_SHADOW', layer=self.layer_id, m=x.shape[0], hash=bool(self.is_hash),
        finite=bool(torch.isfinite(a).all()), rel_rms=float(d.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)),
        tok_rel_max=float(tok.max()), max_abs=float(d.abs().max()), ref_rms=float(b.square().mean().sqrt()),
        equal_frac=float((a == b).float().mean()))


def dual(self, hidden_states, gemm_output_zero_allocator=None, input_ids=None, input_ids_global=None):
    if not eligible(self, hidden_states):
        record('original_dual', self.layer_id, hidden_states.shape[0])
        return ORIG_DUAL(self, hidden_states, gemm_output_zero_allocator, input_ids, input_ids_global=input_ids_global)
    record('tp4moe2_dual', self.layer_id, hidden_states.shape[0])
    out = candidate(self, hidden_states, gemm_output_zero_allocator, input_ids_global, use_alt=self.alt_stream is not None)
    if SHADOW:
        shadow(self, hidden_states, out, lambda: ORIG_DUAL(self, hidden_states, gemm_output_zero_allocator, input_ids,
                                                           input_ids_global=input_ids_global))
    return out


def normal(self, hidden_states, gemm_output_zero_allocator=None, input_ids=None, input_ids_global=None, skip_shared_experts=False):
    if not eligible(self, hidden_states, skip_shared_experts):
        record('original_normal', self.layer_id, hidden_states.shape[0])
        return ORIG_NORMAL(self, hidden_states, gemm_output_zero_allocator, input_ids, input_ids_global=input_ids_global,
                           skip_shared_experts=skip_shared_experts)
    record('tp4moe2_normal', self.layer_id, hidden_states.shape[0])
    out = candidate(self, hidden_states, gemm_output_zero_allocator, input_ids_global, use_alt=False)
    if SHADOW:
        shadow(self, hidden_states, out, lambda: ORIG_NORMAL(self, hidden_states, gemm_output_zero_allocator, input_ids,
                                                             input_ids_global=input_ids_global))
    return out


from sglang.jit_kernel.dsv4 import linear_bf16_fp32   # noqa: E402
G['linear'] = linear_bf16_fp32
Mxfp4HummingMoEMethod.process_weights_after_loading = prepare_weights
D2.DeepseekV2MoE.forward_normal_dual_stream = dual
D2.DeepseekV2MoE.forward_normal = normal
log('FAC_TP4_INSTALLED', variant=FROZEN['variant'], sources=FROZEN['sha256'], shadow=SHADOW, tp2_pdl=os.environ.get('TP2_PDL'),
    sh_fused_maxm=SH_FUSED_MAXM, sh_fused_fromm=SH_FUSED_FROMM)
