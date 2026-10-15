"""TP4 test of the v26 serving hook itself (install_tp4moe2.candidate on a stand-in DeepseekV2MoE), 4 ranks (GPUs of the container).
The two extensions ('sh' fused shared expert, 'ns' shared code compiled out) share one IPC-linked RankState per rank, as in serving.
  1. eager, isolated (sync after every call): hybrid (sh for M <= 32) vs all-ns (alt-stream shared) vs all-sh (fused) -- the builds
     agree to the fused shared expert's summation order; vs production A (Humming + alt shared + custom all-reduce): check_candidate's
     gate (rel_rms < 0.05, per-token < 0.25).
  2. eager, mixed back to back (no sync; sh and ns calls alternate on one workspace) == isolated.
  3. one CUDA graph of the mixed sequence (alt-stream fork/join inside), replayed == isolated.
  4. timing (cold, `copies` layer copies rotated in one graph, worst rank median): A, H (hybrid), N (all-ns), S (all-sh)."""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRV = Path(__file__).resolve().parent
ROOT = SRV.parents[1]
CAP = ROOT / 'diagnostics/real_capture_v2'
MODEL = Path('/models/DeepSeek-V4-Flash-0731')
ARMS = ['A', 'H', 'N', 'S']


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def gate_stats(a, b):
    d = a.float() - b.float()
    tok = d.square().mean(1).sqrt() / b.float().square().mean(1).sqrt().clamp_min(1e-30)
    return float(d.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-30)), float(tok.max())


class Gate:   # MoEGate stand-in: gate(x, alloc) -> fp32 logits; .weight, .e_score_correction_bias.data
    def __init__(self, weight, bias, fn):
        self.weight, self.e_score_correction_bias, self.fn = weight, SimpleNamespace(data=bias), fn

    def __call__(self, x, alloc=None):
        return self.fn(x)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def worker(rank, args):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda', rank)
    from sglang.srt.distributed.parallel_state import init_distributed_environment
    init_distributed_environment(world_size=4, rank=rank, local_rank=rank,
                                 distributed_init_method='file://' + str(args.out / 'rendezvous'), backend='gloo', timeout=600)
    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
    from sglang.srt.distributed.device_communicators.custom_all_reduce import dispatch_custom_allreduce
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
    from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
    from sglang.jit_kernel.dsv4.gemm import linear_bf16_fp32
    from sglang.jit_kernel.dsv4.moe import hash_topk, silu_and_mul_clamp
    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
    from sglang.srt.layers import deep_gemm_wrapper
    from safetensors import safe_open
    os.environ['FAC_FMOE_MODE'] = 'tp4moe2'
    os.environ['FAC_TP4_SHADOW'] = '0'
    I = load_module('install_tp4moe2', SRV / 'install_tp4moe2.py')   # the hook as served (sources.json hashes asserted)
    E = I.ext()
    T = I.T

    group = dist.group.WORLD
    ar = dispatch_custom_allreduce(group, dev)(group, dev)
    assert not ar.disabled

    def gather(obj):
        res = [None] * 4
        dist.all_gather_object(res, obj)
        return res

    lib = CudaRTLibrary()
    keep = []
    st = T.RankState(dev, alloc_comm=False)
    tables = []
    for nbytes in [T.RankState.RS_BYTES(), T.RankState.AG_BYTES()]:
        ptr = lib.cudaMalloc(nbytes)
        lib.cudaMemset(ptr, 0, nbytes)
        keep.append(ptr)
        handle = lib.cudaIpcGetMemHandle(ptr)
        table = []
        for peer, raw in enumerate(gather(bytes(handle))):
            if peer == rank:
                table.append(ptr.value)
            else:
                p = lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(raw))
                keep.append(p)
                table.append(p.value)
        tables.append(table)
    st.link_ptrs(*tables)
    I.G['state'], I.G['rank'] = st, rank
    alt = torch.cuda.Stream()
    report = dict(rank=rank, complete=False, tp2_pdl=os.environ.get('TP2_PDL'), maxm=int(E.MAXM), cases=[],
                  extensions={k: hashlib.sha256(Path(m._ext.__file__).read_bytes()).hexdigest() for k, m in [('ns', I.T), ('sh', I.TS)]})
    target = args.out / f'rank-{rank}.json'

    def save():
        target.write_text(json.dumps(report, indent=1) + '\n')

    wm = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']

    def ck(key):
        with safe_open(MODEL / wm[key], framework='pt', device='cpu') as f:
            return f.get_tensor(key).to(dev)

    config = MoeRunnerConfig(num_experts=256, num_local_experts=256, hidden_size=4096, intermediate_size_per_partition=512,
                             top_k=6, params_dtype=torch.bfloat16, activation='silu', swiglu_limit=10., routed_scaling_factor=1.5)
    for spec in args.modules:
        module, index = spec.split(':')
        index = int(index)
        timed = spec in args.time_modules
        ncp = args.copies if timed else 1
        record = torch.load(CAP / (module + '-weights.pt'), map_location='cpu', weights_only=False)
        metas = record.pop('humming_metas')
        w1, w3, w2 = (ck(f'layers.{index}.ffn.shared_experts.{n}.weight') for n in ['w1', 'w3', 'w2'])
        s1, s3, s2 = (ck(f'layers.{index}.ffn.shared_experts.{n}.scale').float() for n in ['w1', 'w3', 'w2'])
        r = rank
        gu_w0 = torch.cat([w1[512 * r:512 * (r + 1)], w3[512 * r:512 * (r + 1)]]).contiguous()
        gu_s0 = torch.cat([s1[4 * r:4 * (r + 1)], s3[4 * r:4 * (r + 1)]]).contiguous()
        dn_w0 = w2[:, 512 * r:512 * (r + 1)].contiguous()
        dn_s0 = s2[:, 4 * r:4 * (r + 1)].contiguous()
        del w1, w3, w2
        router = ck(f'layers.{index}.ffn.gate.weight').bfloat16().contiguous()
        bias = ck(f'layers.{index}.ffn.gate.bias').float().contiguous() if index >= 3 else None
        ht = ck(f'layers.{index}.ffn.gate.tid2eid').int().contiguous() if index < 3 else None
        layers, hum, moes = [], [], []
        for k in range(ncp):
            layer = torch.nn.Module()
            layer.hidden_size, layer.intermediate_size_per_partition, layer.params_dtype = 4096, 512, torch.bfloat16
            layer.humming_metas = metas
            layer.register_buffer('locks', torch.zeros(1024, dtype=torch.int32, device=dev))
            for key, value in record.items():
                layer.register_buffer(key, value.to(dev).clone())
            layers.append(layer)
            h = HummingRunnerCore(config)
            h.layer = layer
            hum.append(h)
            sh = dict(gu_w=gu_w0.clone(), gu_s=gu_s0.clone(), dn_w=dn_w0.clone(), dn_s=dn_s0.clone())

            def prod_shared(x, alloc=None, sh=sh):   # DeepseekV2MLP (fp8 block quant, swiglu_limit 10) as on sm90
                m = x.shape[0]
                q, s = sglang_per_token_group_quant_fp8(x, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=False)
                gu = torch.empty((m, 1024), dtype=torch.bfloat16, device=dev)
                deep_gemm_wrapper.gemm_nt_f8f8bf16((q, s), (sh['gu_w'], sh['gu_s']), gu)
                act = torch.empty((m, 512), dtype=torch.bfloat16, device=dev)
                silu_and_mul_clamp(gu, act, 10.0)
                q2, s2_ = sglang_per_token_group_quant_fp8(act, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=False)
                y = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
                deep_gemm_wrapper.gemm_nt_f8f8bf16((q2, s2_), (sh['dn_w'], sh['dn_s']), y)
                return y

            def logits_of(x, alloc=None):
                return dsv3_router_gemm(x, router, out_dtype=torch.float32) if x.shape[0] <= 16 else linear_bf16_fp32(x, router)

            def topk(x, logits, expert_location_dispatch_info=None, input_ids=None):
                p, i = hash_topk(logits, input_ids, ht)
                return SimpleNamespace(topk_weights=p, topk_ids=i)

            moe = SimpleNamespace(experts=SimpleNamespace(_fac_tp4_w=T.humming_layer_weights(layer)), alt_stream=alt, is_hash=ht is not None,
                                  gate=Gate(router, bias, logits_of), topk=topk, routed_scaling_factor=1.5, _forward_shared_experts=prod_shared,
                                  layer_id=index, sh=sh)
            moe._fac_sh = dict(w1=sh['gu_w'].view(torch.uint8), s1=sh['gu_s'], w2=sh['dn_w'].view(torch.uint8), s2=sh['dn_s'])
            moes.append(moe)
        del record
        torch.cuda.synchronize()
        dist.barrier()
        rows = []
        for label in ['c8', 'c4', 'c1']:
            batch = torch.load(CAP / (label + '.pt'), map_location='cpu', weights_only=False)
            rows += [v['x'] for (key, _), v in batch.items() if key == module]
            del batch
        pool = torch.cat(rows).to(dev).bfloat16().contiguous()
        while pool.shape[0] < max(args.ms + args.time_ms):
            pool = torch.cat([pool, (pool * 1.01).bfloat16()])
        xs = {m: pool[:m].contiguous() for m in sorted(set(args.ms + args.time_ms))}
        tok = {m: ((torch.arange(m, device=dev, dtype=torch.int64) * 7919 + 12345) % ht.shape[0]) if ht is not None else None for m in xs}

        def prod(k, m):   # forward_normal_dual_stream: routed on main, shared on alt, join, add, all-reduce
            x = xs[m]
            cur = torch.cuda.current_stream()
            alt.wait_stream(cur)
            lg = moes[k].gate(x)
            if ht is not None:
                p, i = hash_topk(lg, tok[m], ht)
            else:
                p, i = moe_fused_gate(lg, bias, topk=6, scoring_func='sqrtsoftplus', num_fused_shared_experts=0, renormalize=True,
                                      routed_scaling_factor=1.5, apply_routed_scaling_factor_on_output=False)
            v = hum[k]._run_indexed_gemm(x, p, i)
            with torch.cuda.stream(alt):
                s = moes[k]._forward_shared_experts(x)
            cur.wait_stream(alt)
            return ar.custom_all_reduce(v + s)

        def cand(k, m, maxm):
            I.SH_FUSED_MAXM = maxm
            return I.candidate(moes[k], xs[m], None, tok[m], use_alt=True)

        HYB, NS, SH = 32, 0, int(E.MAXM)
        row = dict(module=module, hash_mode=ht is not None, iso={}, mixed={}, graph={})
        report['cases'].append(row)
        iso = {}
        for m in args.ms:   # 1. isolated
            dist.barrier()
            res = {}
            for lab, maxm in [('H', HYB), ('N', NS), ('S', SH)]:
                for _ in range(2):
                    v = cand(0, m, maxm)
                torch.cuda.synchronize()
                res[lab] = v.clone()
            a = prod(0, m)
            torch.cuda.synchronize()
            rr, tr = gate_stats(res['H'], a)
            c = dict(h_vs_n=rel(res['H'], res['N']), h_vs_s=rel(res['H'], res['S']), rel_rms_vs_A=rr, tok_rel_max_vs_A=tr,
                     finite=all(bool(torch.isfinite(t.float()).all()) for t in res.values()))
            row['iso'][m] = c
            c['s_ok'] = c['h_vs_s'] < 2e-3   # all-sh above 32 is not served (recorded, not gated)
            ok = c['finite'] and c['h_vs_n'] < 2e-3 and (c['s_ok'] or m > HYB) and rr < 0.05 and tr < 0.25
            assert ok, (module, m, c)
            iso[m] = res['H']
        save()
        for rep in range(2):   # 2. mixed back to back
            dist.barrier()
            outs = [(m, cand(0, m, HYB)) for m in args.ms]
            torch.cuda.synchronize()
            for m, v in outs:
                row['mixed'].setdefault(m, []).append(dict(rel=rel(v, iso[m]), equal=float((v == iso[m]).float().mean())))
                assert rel(v, iso[m]) < 1e-3, (module, m, rep, rel(v, iso[m]))
        g = torch.cuda.CUDAGraph()   # 3. one graph of the mixed sequence
        with torch.cuda.graph(g):
            gouts = [(m, cand(0, m, HYB)) for m in args.ms]
        for rep in range(3):
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            for m, v in gouts:
                row['graph'].setdefault(m, []).append(dict(rel=rel(v, iso[m]), equal=float((v == iso[m]).float().mean())))
                assert rel(v, iso[m]) < 1e-3, (module, m, rep, rel(v, iso[m]))
        del g, gouts
        save()
        if rank == 0:
            print('CHECK', module, json.dumps({m: dict(h_vs_n=round(c['h_vs_n'], 6), h_vs_s=round(c['h_vs_s'], 6), rr=round(c['rel_rms_vs_A'], 4),
                                                       eq_mixed=min(e['equal'] for e in row['mixed'][m]),
                                                       eq_graph=min(e['equal'] for e in row['graph'][m])) for m, c in row['iso'].items()}), flush=True)
        if timed:   # 4. timing
            seq = [k % ncp for k in range(2 * ncp)]
            row['timing'] = {}
            for m in args.time_ms:
                graphs = {}
                for lab in ARMS:
                    g = torch.cuda.CUDAGraph()
                    for k in range(ncp):   # warm (eager) once per copy: maps / JIT
                        (prod(k, m) if lab == 'A' else cand(k, m, dict(H=HYB, N=NS, S=SH)[lab]))
                    torch.cuda.synchronize()
                    dist.barrier()
                    if lab == 'A':
                        with ar.capture():
                            with torch.cuda.graph(g):
                                for k in seq:
                                    prod(k, m)
                    else:
                        with torch.cuda.graph(g):
                            for k in seq:
                                cand(k, m, dict(H=HYB, N=NS, S=SH)[lab])
                    graphs[lab] = g
                    dist.barrier()
                    for _ in range(8):
                        g.replay()
                    torch.cuda.synchronize()
                trials = []
                for lab in ARMS + ARMS:
                    dist.barrier()
                    g = graphs[lab]
                    for _ in range(10):
                        g.replay()
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(7):
                        dist.barrier()
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        for _ in range(20):
                            g.replay()
                        e.record()
                        e.synchronize()
                        samples.append(s.elapsed_time(e) * 1e3 / (20 * len(seq)))
                    byrank = gather(samples)
                    worst = [max(byrank[q][j] for q in range(4)) for j in range(7)]
                    trials.append((lab, statistics.median(worst)))
                med = {lab: statistics.median([t for l2, t in trials if l2 == lab]) for lab in ARMS}
                row['timing'][m] = med
                save()
                if rank == 0:
                    print('TIME', module, m, json.dumps({k: round(v, 2) for k, v in med.items()}),
                          'A/H %.3f A/N %.3f A/S %.3f' % (med['A'] / med['H'], med['A'] / med['N'], med['A'] / med['S']), flush=True)
                del graphs
                dist.barrier()
        del layers, hum, moes
        torch.cuda.empty_cache()
    report['complete'] = True
    save()
    dist.barrier()
    ar.close()
    ar.disabled = True
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modules', nargs='+', default=['layer-20-module-20:20', 'layer-0-module-0:0'])
    ap.add_argument('--time-modules', nargs='*', default=['layer-20-module-20:20'])
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 64, 6, 128, 32, 33, 16, 96, 2, 48, 12, 30, 384, 24])
    ap.add_argument('--time-ms', type=int, nargs='*', default=[1, 2, 4, 6, 8, 12, 16, 24, 30, 32, 36, 48, 64, 96, 128, 192, 256, 384])
    ap.add_argument('--copies', type=int, default=6)
    args = ap.parse_args()
    assert torch.cuda.device_count() == 4
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'source.json').write_text(json.dumps({'test_v26_mix.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'argv': sys.argv,
                                                      'env_TP2_PDL': os.environ.get('TP2_PDL')}, indent=1) + '\n')
    mp.spawn(worker, args=(args,), nprocs=4, join=True)
    results = [json.loads((args.out / f'rank-{r}.json').read_text()) for r in range(4)]
    assert all(r['complete'] for r in results)
    print('V26_MIX_COMPLETE')
