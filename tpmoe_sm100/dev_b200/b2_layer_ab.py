"""Single-layer timing on B200 at TP4 (one process per GPU), real rank shards of DeepSeek-V4-Flash layer 20.

Arms (per M, CUDA graphs rotating `--copies` weight copies = cold L2, worst-rank median over samples, arm order shuffled per round):
  A    production layer (b2_common.Prod: router GEMM + moe_fused_gate + trtllm-gen MXFP4 routed MoE on the main stream, DeepGEMM
       FP8 shared expert on the alt stream, shared + 1.5 routed, custom all-reduce)
  RT   router logits alone;  TK  moe_fused_gate alone;  RM  routed alone (pack + mxfp8_quantize + trtllm-gen, given top-k)
  SH   the shared expert alone;  AR  the custom all-reduce of a BF16 [M, 4096] alone
  N    tpmoe_b2 layer: router logits + the shared expert on the alt stream, join, the fused kernel (gate, experts, x 1.5 + shared, LL AR)
  KN   the fused kernel alone (precomputed logits and shared output)
Correctness on the way (rank-local): RM vs the fp32 reference (b2_ref.routed_ref) with production's own MXFP8 x and with the
reference's MXFP8 x.  usage (serving image, 4 GPUs):
  python3 b2_layer_ab.py --ndev 4 --model MODEL --cap CAPDIR --out OUT"""
import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))


def worker(rank, args):
    ndev = args.ndev
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    from sglang.srt.distributed.parallel_state import init_distributed_environment
    init_distributed_environment(world_size=ndev, rank=rank, local_rank=rank,
                                 distributed_init_method='file://' + str(args.out / 'rendezvous'), backend='gloo', timeout=900)
    from sglang.srt.distributed.device_communicators.custom_all_reduce import dispatch_custom_allreduce
    import b2_common as C
    import b2_ref as R

    group = dist.group.WORLD
    ar = dispatch_custom_allreduce(group, dev)(group, dev)
    assert not ar.disabled

    def gather(obj):
        res = [None] * ndev
        dist.all_gather_object(res, obj)
        return res

    wm = C.weight_map(args.model)
    raw = C.load_experts_raw(args.model, wm, args.layer, ndev, rank, dev)
    shared = C.load_shared(args.model, wm, args.layer, ndev, rank, dev)
    router, bias = C.load_router(args.model, wm, args.layer, dev)
    alt = torch.cuda.Stream()
    import tpb2
    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
    KE = tpb2.load_ext(tag=('_' + args.flags.replace('-D', '').replace('=', '').replace(' ', '_')) if args.flags else '', flags=tuple(args.flags.split()))
    st = tpb2.RankState(KE, dev, ndev, rank)
    lib = CudaRTLibrary()
    keep, tables = [], []
    for nbytes in [tpb2.RankState.RS_BYTES(KE), tpb2.RankState.AG_BYTES(KE)]:
        ptr = lib.cudaMalloc(nbytes)
        lib.cudaMemset(ptr, 0, nbytes)
        keep.append(ptr)
        handle = lib.cudaIpcGetMemHandle(ptr)
        table = []
        for peer, rawh in enumerate(gather(bytes(handle))):
            if peer == rank:
                table.append(ptr.value)
            else:
                pp = lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(rawh))
                keep.append(pp)
                table.append(pp.value)
        tables.append(table)
    st.link_ptrs(*tables)
    prods = []
    for k in range(args.copies):
        layer = C.sglang_trtllm_layer(raw, dev)
        shk = dict(shared, gu_w=shared['gu_w'].clone(), gu_s=shared['gu_s'].clone(), dn_w=shared['dn_w'].clone(),
                   dn_s=shared['dn_s'].clone())
        prods.append(C.Prod(layer, shk, router, bias, alt, ar))
        prods[-1].kw = tpb2.Weights(KE, layer.w13_weight, layer.w13_weight_scale_inv, layer.w2_weight, layer.w2_weight_scale_inv)
        prods[-1].sw = tpb2.SharedWeights(KE, shk['gu_w'], shared['gs_f'], shk['dn_w'], shared['ds_f'])

    def ours(k, m, with_shared=True):   # router logits + shared on the alt stream, join, the fused kernel
        x = xs[m]
        Pk = prods[k]
        cur = torch.cuda.current_stream()
        alt.wait_stream(cur)
        lg = Pk.logits(x)
        with torch.cuda.stream(alt):
            s = Pk.shared(x)
        cur.wait_stream(alt)
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, Pk.kw, x, out, glog=lg, gbias=bias, shared=s)

    def ours_sh(k, m):   # router logits, the fused kernel with the shared expert inside (no alt stream)
        Pk = prods[k]
        lg = Pk.logits(xs[m])
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, Pk.kw, xs[m], out, glog=lg, gbias=bias, SW=Pk.sw)

    def ours_sh_ker(k, m):
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, prods[k].kw, xs[m], out, glog=pre[m][0], gbias=bias, SW=prods[k].sw)

    def ours_ker(k, m):
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, prods[k].kw, xs[m], out, glog=pre[m][0], gbias=bias, shared=pre[m][4])
    xs = C.capture_pool(args.cap, args.ms, dev)
    torch.cuda.synchronize()
    dist.barrier()
    report = dict(rank=rank, ar=type(ar).__name__, gpu=torch.cuda.get_device_name(dev), timing={}, check={})
    if rank == 0:
        print('AR', report['ar'], 'GPU', report['gpu'], flush=True)
    pre = {}
    arms = {
        'A': lambda k, m: prods[k](xs[m]),
        'RT': lambda k, m: prods[k].logits(xs[m]),
        'TK': lambda k, m: prods[k].topk(pre[m][0]),
        'RM': lambda k, m: prods[k].routed(xs[m], pre[m][1], pre[m][2]),
        'SH': lambda k, m: prods[k].shared(xs[m]),
        'AR': lambda k, m: ar.custom_all_reduce(pre[m][3]),
        'N': lambda k, m: ours(k, m),
        'KN': lambda k, m: ours_ker(k, m),
        'S': lambda k, m: ours_sh(k, m),
        'KS': lambda k, m: ours_sh_ker(k, m),
    }
    if args.arms:
        arms = {a: arms[a] for a in args.arms}
    ncp = args.copies
    seq = [k % ncp for k in range(2 * ncp)]
    rnd = random.Random(1234)
    for m in args.ms:
        P = prods[0]
        lg = P.logits(xs[m])
        w, i = P.topk(lg)
        pre[m] = (lg, w, i, P(xs[m], with_ar=False).clone(), P.shared(xs[m]).clone())
        # correctness: the routed output vs the fp32 reference
        from flashinfer import mxfp8_quantize
        r = P.routed(xs[m], w, i).clone()
        xq, xsf = mxfp8_quantize(xs[m], False, alignment=C.DIM, backend='cute-dsl')
        xs_e = xsf.view(torch.uint8).reshape(m, -1)[:, :C.DIM // 32].to(torch.int32)
        xq_d = xq.view(torch.float8_e4m3fn).float().reshape(m, C.DIM // 32, 32) * torch.exp2(xs_e.float() - 127).unsqueeze(-1)
        ref_fi = R.routed_ref(xs[m], i, w, raw, xq=xq_d.reshape(m, C.DIM))
        ref_own = R.routed_ref(xs[m], i, w, raw)
        ref_b16 = R.routed_ref(xs[m], i, w, raw, xq=xq_d.reshape(m, C.DIM), act_bf16=True)
        xq_own = R.quant_mxfp8(xs[m])[0]
        # the fused kernel vs production (full layer, after the all-reduce) and vs the reference (all-reduced in fp32 here)
        a_full = P(xs[m]).clone()
        o_full = ours(0, m).clone()
        s_full = ours_sh(0, m).clone()
        torch.cuda.synchronize()
        gate_eq = bool(torch.equal(st.ids[:m * 6].view(m, 6), i.to(torch.int32))) and bool(torch.equal(st.rw[:m * 6].view(m, 6), w.float()))
        loc = (pre[m][4].float() + 1.5 * ref_fi.float()).bfloat16().float().cpu()
        dist.all_reduce(loc)
        ref_full = loc.bfloat16()
        chk = dict(rm_vs_ref_fi=R.rel(r, ref_fi), rm_vs_ref_own=R.rel(r, ref_own), rm_vs_ref_actbf16=R.rel(r, ref_b16),
                   rm_tokmax_ref_fi=R.tok_rel_max(r, ref_fi), xq_fi_eq_own=bool(torch.equal(xq_d.reshape(m, C.DIM), xq_own)),
                   finite=bool(torch.isfinite(r.float()).all()), n_vs_a=R.rel(o_full, a_full), n_vs_ref=R.rel(o_full.cpu(), ref_full),
                   a_vs_ref=R.rel(a_full.cpu(), ref_full), n_tokmax_ref=R.tok_rel_max(o_full.cpu(), ref_full), gate_eq=gate_eq,
                   n_finite=bool(torch.isfinite(o_full.float()).all()), n_eq_a=bool(torch.equal(o_full, a_full)),
                   n_ne_a=int((o_full != a_full).sum()), s_ne_a=int((s_full != a_full).sum()), s_vs_a=R.rel(s_full, a_full))
        report['check'][m] = chk
        if args.rounds == 0:   # correctness only
            allchk = gather(chk)
            if rank == 0:
                print(f'TP{ndev} M {m:4d} check: N vs A {max(c["n_vs_a"] for c in allchk):.5f}  N vs ref {max(c["n_vs_ref"] for c in allchk):.5f}  '
                      f'A vs ref {max(c["a_vs_ref"] for c in allchk):.5f}  N tokmax ref {max(c["n_tokmax_ref"] for c in allchk):.4f}  '
                      f'gate==prod {all(c["gate_eq"] for c in allchk)}  finite {all(c["n_finite"] for c in allchk)}  '
                      f'bitwise N==A {[c["n_eq_a"] for c in allchk]} mismatches {[c["n_ne_a"] for c in allchk]}  '
                  f'S vs A {max(c["s_vs_a"] for c in allchk):.5f} mismatches {[c["s_ne_a"] for c in allchk]}', flush=True)
            continue
        graphs = {}
        for lab, fn in arms.items():
            for k in range(ncp):
                fn(k, m)
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            if lab in ('A', 'AR', 'N', 'KN', 'S', 'KS'):
                with ar.capture():
                    with torch.cuda.graph(g):
                        for k in seq:
                            fn(k, m)
            else:
                with torch.cuda.graph(g):
                    for k in seq:
                        fn(k, m)
            graphs[lab] = g
            dist.barrier()
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
        samples = {lab: [] for lab in arms}
        for _ in range(args.rounds):
            order = list(arms)
            rnd.shuffle(order)
            for lab in order:
                g = graphs[lab]
                dist.barrier()
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                dist.barrier()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    g.replay()
                e.record()
                e.synchronize()
                t = s.elapsed_time(e) * 1e3 / (10 * len(seq))
                samples[lab].append(max(gather(t)))   # worst rank
        med = {lab: statistics.median(v) for lab, v in samples.items()}
        report['timing'][m] = med
        allchk = gather(chk)
        if rank == 0:
            t = med
            print(f'TP{ndev} M {m:4d}: ' + '  '.join(f'{lab} {t[lab]:7.1f}' for lab in arms) + ' | ' +
                  f'RM vs ref(fi x) max-rank {max(c["rm_vs_ref_fi"] for c in allchk):.5f}  (own x) {max(c["rm_vs_ref_own"] for c in allchk):.5f}  '
                  f'(act bf16) {max(c["rm_vs_ref_actbf16"] for c in allchk):.5f}  tokmax {max(c["rm_tokmax_ref_fi"] for c in allchk):.4f}  '
                  f'xq fi==own {all(c["xq_fi_eq_own"] for c in allchk)}  finite {all(c["finite"] for c in allchk)}', flush=True)
            print(f'      ours: N vs A {max(c["n_vs_a"] for c in allchk):.5f}  N vs ref {max(c["n_vs_ref"] for c in allchk):.5f}  '
                  f'A vs ref {max(c["a_vs_ref"] for c in allchk):.5f}  N tokmax ref {max(c["n_tokmax_ref"] for c in allchk):.4f}  '
                  f'gate==prod {all(c["gate_eq"] for c in allchk)}  finite {all(c["n_finite"] for c in allchk)}  '
                  f'bitwise N==A {[c["n_eq_a"] for c in allchk]} mismatches {[c["n_ne_a"] for c in allchk]}  '
                  f'S vs A {max(c["s_vs_a"] for c in allchk):.5f} mismatches {[c["s_ne_a"] for c in allchk]}', flush=True)
        (args.out / f'rank-{rank}.json').write_text(json.dumps(report, indent=1) + '\n')
        del graphs
        dist.barrier()
    dist.barrier()
    ar.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ndev', type=int, default=4)
    ap.add_argument('--model', required=True)
    ap.add_argument('--cap', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layer', type=int, default=20)
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 2, 4, 6, 8, 16, 20, 24, 32, 48, 64, 96, 128])
    ap.add_argument('--copies', type=int, default=4)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--arms', nargs='*')
    ap.add_argument('--flags', default='', help='extra nvcc -D flags for the fused kernel build (own build dir)')
    args = ap.parse_args()
    assert torch.cuda.device_count() >= args.ndev
    args.out.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(args,), nprocs=args.ndev, join=True)
    print('B2_LAYER_AB_COMPLETE')
