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
    ncg = dist.new_group(backend='nccl') if args.gpu_align else None   # (--gpu-align: GPU-side rank alignment right before each timed region)
    tiny = torch.zeros(16, dtype=torch.float32, device=dev)

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
    args.flags = args.flags.replace(',', ' ')   # (comma-separated on the command line: run_b2.sh re-splits spaces)
    for f_ in ('flags', 'flags2', 'flags3', 'xbuilds'):   # '@R' in build flags: this rank (per-rank builds, e.g. -DTPB2_RANK=@R)
        if getattr(args, f_, None): setattr(args, f_, getattr(args, f_).replace('@R', str(rank)))
    if args.stamps:
        args.flags = (args.flags + ' -DTPB2_DBG=1').strip()
    if args.trace:
        args.flags = (args.flags + f' -DTPB2_TRACE={args.xt_fmt}' + (f' -DTPB2_XT_MASK={args.xt_mask}' if args.xt_fmt == 2 else '')).strip()
    tagf = lambda f: ('_' + f.replace('-D', '').replace('=', '').replace(' ', '_')) if f else ''
    KE = tpb2.load_ext(tag=tagf(args.flags), flags=tuple(args.flags.split()))
    lib = CudaRTLibrary()
    keep = []

    def make_state(E):   # workspace + IPC-linked all-reduce buffers of one kernel build
        st_ = tpb2.RankState(E, dev, ndev, rank)
        tables = []
        if args.mcbuf and ndev > 1:   # symmetric-memory buffers (cuMem VMM, peer-mapped) + their NVLS multicast address as entry ndev
            import torch.distributed._symmetric_memory as symm_mem
            for nbytes in [tpb2.RankState.RS_BYTES(E), tpb2.RankState.AG_BYTES(E)]:
                t = symm_mem.empty(nbytes, dtype=torch.uint8, device=dev)
                t.zero_()
                h = symm_mem.rendezvous(t, group.group_name)
                keep.append((t, h))
                off = t.data_ptr() - h.buffer_ptrs[rank]
                assert h.multicast_ptr != 0, 'no multicast support for the symmetric buffers'
                tables.append([h.buffer_ptrs[d] + off for d in range(ndev)] + [h.multicast_ptr + off])
            torch.cuda.synchronize()
            dist.barrier()
            st_.link_ptrs(*tables)
            return st_
        for nbytes in [tpb2.RankState.RS_BYTES(E), tpb2.RankState.AG_BYTES(E)]:
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
        st_.link_ptrs(*tables)
        return st_
    st = make_state(KE)
    KE2 = st2 = None
    if args.flags2 is not None:   # a second build of the kernel, timed in the same rounds (arm KS2): paired A/B
        args.flags2 = args.flags2.replace(',', ' ')
        KE2 = tpb2.load_ext(src=args.src2, tag=tagf(args.flags2) + '_b' + ('' if args.src2 == 'tpmoe_b2.cu' else '_' + args.src2.replace('.cu', '')),
                            flags=tuple(args.flags2.split()))
        st2 = make_state(KE2)
    KE3 = st3 = None
    if args.flags3 is not None:   # a third build (arm D / DF: per-M dispatch over builds 1..3)
        args.flags3 = args.flags3.replace(',', ' ')
        KE3 = tpb2.load_ext(tag=tagf(args.flags3) + '_c', flags=tuple(args.flags3.split()))
        st3 = make_state(KE3)
    KX, stx = [], []   # --xbuilds: builds 4, 5, .. (per-M dispatch)
    for i, f in enumerate(args.xbuilds.split('+') if args.xbuilds else []):
        f = f.replace(',', ' ').strip()
        KX.append(tpb2.load_ext(tag=tagf(f) + '_x', flags=tuple(f.split())))
        stx.append(make_state(KX[-1]))
    prods = []
    for k in range(args.copies):
        layer = C.sglang_trtllm_layer(raw, dev)
        shk = dict(shared, gu_w=shared['gu_w'].clone(), gu_s=shared['gu_s'].clone(), dn_w=shared['dn_w'].clone(),
                   dn_s=shared['dn_s'].clone())
        prods.append(C.Prod(layer, shk, router, bias, alt, ar))
        prods[-1].kw = tpb2.Weights(KE, layer.w13_weight, layer.w13_weight_scale_inv, layer.w2_weight, layer.w2_weight_scale_inv)
        prods[-1].sw = tpb2.SharedWeights(KE, shk['gu_w'], shared['gs_f'], shk['dn_w'], shared['ds_f'])
        if KE2 is not None:
            prods[-1].kw2 = tpb2.Weights(KE2, layer.w13_weight, layer.w13_weight_scale_inv, layer.w2_weight, layer.w2_weight_scale_inv)
            prods[-1].sw2 = tpb2.SharedWeights(KE2, shk['gu_w'], shared['gs_f'], shk['dn_w'], shared['ds_f'])
        if KE3 is not None:
            prods[-1].kw3 = tpb2.Weights(KE3, layer.w13_weight, layer.w13_weight_scale_inv, layer.w2_weight, layer.w2_weight_scale_inv)
            prods[-1].sw3 = tpb2.SharedWeights(KE3, shk['gu_w'], shared['gs_f'], shk['dn_w'], shared['ds_f'])
        prods[-1].kwx = [tpb2.Weights(E_, layer.w13_weight, layer.w13_weight_scale_inv, layer.w2_weight, layer.w2_weight_scale_inv) for E_ in KX]
        prods[-1].swx = [tpb2.SharedWeights(E_, shk['gu_w'], shared['gs_f'], shk['dn_w'], shared['ds_f']) for E_ in KX]

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

    RF = None
    if args.rfuse:   # our router GEMM (b2_rfuse.cu: bitwise == cuBLAS K-parts) for 5 <= M <= 64, PDL-overlapped with the MoE kernel
        from torch.utils.cpp_extension import load as _load
        from router_splits import router_parts
        os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
        (SRV / 'build_rfuse').mkdir(exist_ok=True)
        RF = _load(name='b2_rfuse', sources=[str(SRV / 'b2_rfuse.cu')], build_directory=str(SRV / 'build_rfuse'),
                   extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'],
                   extra_ldflags=['-lcuda'], verbose=False)
        rparts = {}

    def ours_shr_parts(m):
        if m not in rparts:
            sp = router_parts(m)
            b_, k_ = [], 0
            for kn in sp:
                b_ += [k_, kn]; k_ += kn
            rparts[m] = (b_, len(sp), torch.empty(len(sp), m, 256, dtype=torch.float32, device=dev))

    def ours_shr(k, m):   # our router (5 <= M <= 64; else production's) + the fused kernel with the shared expert inside
        Pk = prods[k]
        if RF is None or not (5 <= m <= args.rfmax):
            return ours_sh(k, m)
        ours_shr_parts(m)
        b_, S_, part_ = rparts[m]
        RF.run(xs[m], Pk.router, part_, b_, None, 1)
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, Pk.kw, xs[m], out, glog=part_, gbias=bias, SW=Pk.sw, nparts=S_)

    def our_router(k, m):   # arm RR: our router GEMM alone (5 <= M <= 64), cf. RT
        ours_shr_parts(m)
        b_, S_, part_ = rparts[m]
        RF.run(xs[m], prods[k].router, part_, b_, None, 1)
        return part_

    rfp = {}

    def ours_shd(k, m):   # arm SD (NOT bitwise for M >= 5): sglang dsv3_router_gemm (FMA chains) as the router at every M <= 16
        from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
        lg = dsv3_router_gemm(xs[m], prods[k].router, out_dtype=torch.float32) if m <= 16 else prods[k].logits(xs[m])
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, prods[k].kw, xs[m], out, glog=lg, gbias=bias, SW=prods[k].sw)

    def ours_shf(k, m):   # arm SF: the in-kernel router (5 <= M <= 32, K-parts <= 1408) + the fused kernel with the shared expert inside
        from router_splits import router_parts
        sp = router_parts(m)
        if sp == 'dsv3' and m <= 4:   # (M <= 4: production's dsv3_router_gemm reproduced in the kernel)
            if m not in rfp:
                rfp[m] = torch.empty(1, m, 256, dtype=torch.float32, device=dev)
            out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
            return tpb2.run(st, prods[k].kw, xs[m], out, glog=rfp[m], gbias=bias, SW=prods[k].sw, nparts=1, router=(prods[k].router, 'dsv3'))
        if not isinstance(sp, tuple) or not (5 <= m <= args.rfmax) or max(sp) > 2048:   # (M96/128 in-kernel: bitwise but +2..4.6 us -- 512 KB of x per unit)
            return ours_sh(k, m)
        if m not in rfp:
            rfp[m] = torch.empty(len(sp), m, 256, dtype=torch.float32, device=dev)
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, prods[k].kw, xs[m], out, glog=rfp[m], gbias=bias, SW=prods[k].sw, nparts=len(sp), router=(prods[k].router, sp))

    rde = {}

    def ours_she(k, m):   # arm SE (M <= 4): our copy of the dsv3 router (bitwise; W loaded before the PDL wait, dependents triggered early)
        if m > 4:
            return ours_sh(k, m)
        if m not in rde:
            rde[m] = torch.empty(m, 256, dtype=torch.float32, device=dev)
        KE.router_dsv3(xs[m], prods[k].router, rde[m])
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, prods[k].kw, xs[m], out, glog=rde[m], gbias=bias, SW=prods[k].sw)

    def ours_shf2(k, m):   # arm SF2: the in-kernel router + the second build (M-dispatched builds)
        from router_splits import router_parts
        sp = router_parts(m)
        if not isinstance(sp, tuple) or not (5 <= m <= args.rfmax) or max(sp) > 2048:
            return ours_sh2(k, m)
        if m not in rfp:
            rfp[m] = torch.empty(len(sp), m, 256, dtype=torch.float32, device=dev)
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st2, prods[k].kw2, xs[m], out, glog=rfp[m], gbias=bias, SW=prods[k].sw2, nparts=len(sp), router=(prods[k].router, sp))

    disp = [tuple(int(v) for v in e.split(':')) for e in args.dispatch.split(',')] if args.dispatch else []

    def ours_disp(k, m, fused):   # arms D / DF: the build for M by --dispatch "maxM:build,..."; DF: the in-kernel router at M5-64
        from router_splits import router_parts
        bi = next(b_ for mx, b_ in disp if m <= mx)
        Pk = prods[k]
        if bi >= 4:
            st_, kw_, sw_ = stx[bi - 4], Pk.kwx[bi - 4], Pk.swx[bi - 4]
        else:
            st_ = {1: st, 2: st2, 3: st3}[bi]
            kw_, sw_ = {1: (Pk.kw, Pk.sw), 2: (getattr(Pk, 'kw2', None), getattr(Pk, 'sw2', None)), 3: (getattr(Pk, 'kw3', None), getattr(Pk, 'sw3', None))}[bi]
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        sp = router_parts(m)
        if fused and isinstance(sp, tuple) and 5 <= m <= args.rfmax and max(sp) <= 2048:
            if m not in rfp:
                rfp[m] = torch.empty(len(sp), m, 256, dtype=torch.float32, device=dev)
            return tpb2.run(st_, kw_, xs[m], out, glog=rfp[m], gbias=bias, SW=sw_, nparts=len(sp), router=(Pk.router, sp))
        return tpb2.run(st_, kw_, xs[m], out, glog=Pk.logits(xs[m]), gbias=bias, SW=sw_)

    def ours_sh_ker(k, m):
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, prods[k].kw, xs[m], out, glog=pre[m][0], gbias=bias, SW=prods[k].sw)

    def ours_sh2(k, m):   # arm S2: the production router + the second build (paired S A/B)
        Pk = prods[k]
        lg = Pk.logits(xs[m])
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st2, Pk.kw2, xs[m], out, glog=lg, gbias=bias, SW=Pk.sw2)

    def ours_sh_ker2(k, m):
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st2, prods[k].kw2, xs[m], out, glog=pre[m][0], gbias=bias, SW=prods[k].sw2)

    def ours_ker(k, m):
        out = torch.empty((m, C.DIM), dtype=torch.bfloat16, device=dev)
        return tpb2.run(st, prods[k].kw, xs[m], out, glog=pre[m][0], gbias=bias, shared=pre[m][4])
    xs = C.capture_pool(args.cap, args.ms, dev, off=args.xoff)
    torch.cuda.synchronize()
    dist.barrier()
    report = dict(rank=rank, ar=type(ar).__name__, gpu=torch.cuda.get_device_name(dev), timing={}, check={})
    if rank == 0:
        print('AR', report['ar'], 'GPU', report['gpu'], flush=True)
    pre = {}
    arms = {
        'A': lambda k, m: prods[k](xs[m]),
        'RT': lambda k, m: prods[k].logits(xs[m]),
        'RR': lambda k, m: our_router(k, m),
        'TK': lambda k, m: prods[k].topk(pre[m][0]),
        'RM': lambda k, m: prods[k].routed(xs[m], pre[m][1], pre[m][2]),
        'SH': lambda k, m: prods[k].shared(xs[m]),
        'AR': lambda k, m: ar.custom_all_reduce(pre[m][3]),
        'N': lambda k, m: ours(k, m),
        'KN': lambda k, m: ours_ker(k, m),
        'S': lambda k, m: ours_sh(k, m),
        'KS': lambda k, m: ours_sh_ker(k, m),
        'KS2': lambda k, m: ours_sh_ker2(k, m),
        'S2': lambda k, m: ours_sh2(k, m),
        'SE': lambda k, m: ours_she(k, m),
        'SF2': lambda k, m: ours_shf2(k, m),
        'D': lambda k, m: ours_disp(k, m, False),
        'DF': lambda k, m: ours_disp(k, m, True),
        'SR': lambda k, m: ours_shr(k, m),
        'SF': lambda k, m: ours_shf(k, m),
        'SD': lambda k, m: ours_shd(k, m),
    }
    if args.arms:
        arms = {a: arms[a] for a in args.arms}
    ncp = args.copies
    seq = [k % ncp for k in range(2 * ncp)]
    rnd = random.Random(1234)
    if args.sweep:   # bitwise sweep: production layer vs ours (build 1, and build 2 if --flags2) over consecutive capture offsets
        big = C.capture_pool(args.cap, [args.sweep + max(args.ms)], dev)[args.sweep + max(args.ms)]
        for m in args.ms:
            res = [0, 0, 0, 0, 0, 0, 0, 0]   # build 1: offsets with mismatches, elements; build 2: same; build 1 + in-kernel router (--arms SF): same; dispatched (D / DF)
            sf = bool(args.arms and 'SF' in args.arms and 5 <= m <= args.rfmax)
            sd = bool(args.arms and ('D' in args.arms or 'DF' in args.arms))
            for off in range(args.sweep):
                xs[m] = big[off:off + m].contiguous()
                a = prods[0](xs[m]).clone()
                s1 = ours_sh(0, m).clone()
                s2 = ours_sh2(0, m).clone() if st2 is not None else None
                s3 = ours_shf(0, m).clone() if sf else None
                s4 = ours_disp(0, m, 'DF' in args.arms).clone() if sd else None
                torch.cuda.synchronize()
                n1 = int((s1 != a).sum()); res[0] += n1 > 0; res[1] += n1
                if s2 is not None:
                    n2 = int((s2 != a).sum()); res[2] += n2 > 0; res[3] += n2
                if s3 is not None:
                    n3 = int((s3 != a).sum()); res[4] += n3 > 0; res[5] += n3
                if s4 is not None:
                    n4 = int((s4 != a).sum()); res[6] += n4 > 0; res[7] += n4
            allr = gather(res)
            if rank == 0:
                print(f'SWEEP M {m}: {args.sweep} offsets  build1 offsets with mismatches {[r_[0] for r_ in allr]} elements {[r_[1] for r_ in allr]}'
                      + (f'  build2 {[r_[2] for r_ in allr]} elements {[r_[3] for r_ in allr]}' if st2 is not None else '')
                      + (f'  build1+SF {[r_[4] for r_ in allr]} elements {[r_[5] for r_ in allr]}' if sf else '')
                      + (f'  dispatched {[r_[6] for r_ in allr]} elements {[r_[7] for r_ in allr]}' if sd else ''), flush=True)
        if rank == 0:
            print('B2_LAYER_AB_COMPLETE', flush=True)
        return
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
        o_full = (ours_sh(0, m) if 'TPB2_SPEC=1' in (args.flags or '') else ours(0, m)).clone()   # (a SPEC build: fused shared expert only)
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
        if RF is not None:   # our router + kernel vs production router + kernel, bitwise
            srf = ours_shr(0, m).clone()
            torch.cuda.synchronize()
            chk['sr_ne_s'] = int((srf != s_full).sum())
        if args.arms and 'SD' in args.arms:   # dsv3 FMA router (M <= 16) + kernel vs production: how much differs
            sdf = ours_shd(0, m).clone()
            torch.cuda.synchronize()
            chk['sd_ne_s'] = int((sdf != s_full).sum()); chk['sd_vs_s'] = R.rel(sdf, s_full)
        if args.arms and 'SF' in args.arms:   # in-kernel router + kernel vs production router + kernel, bitwise
            sff = ours_shf(0, m).clone()
            torch.cuda.synchronize()
            chk['sf_ne_s'] = int((sff != s_full).sum())
            if m <= 4 and m in rfp:   # (dsv3 in the kernel: the logits themselves, bit for bit)
                lg = prods[0].logits(xs[m]).float()
                torch.cuda.synchronize()
                chk['sf_lg_ne'] = int((rfp[m][0].view_as(lg) != lg).sum())
        if args.arms and 'SE' in args.arms and m <= 4:   # our dsv3 copy: logits and outputs vs production, bitwise
            sef = ours_she(0, m).clone()
            lg0 = prods[0].logits(xs[m]).float()
            torch.cuda.synchronize()
            chk['se_ne_s'] = int((sef != s_full).sum()); chk['se_lg_ne'] = int((rde[m] != lg0).sum())
        if args.arms and ('D' in args.arms or 'DF' in args.arms):   # the dispatched layer vs production, bitwise
            dd = ours_disp(0, m, 'DF' in args.arms).clone()
            torch.cuda.synchronize()
            chk['d_ne_a'] = int((dd != a_full).sum())
        if st2 is not None:   # the second build vs the first, bitwise
            k1 = ours_sh_ker(0, m).clone()
            k2 = ours_sh_ker2(0, m).clone()
            torch.cuda.synchronize()
            chk['ks2_ne_ks'] = int((k1 != k2).sum())
        report['check'][m] = chk
        if args.stamps:   # one cold-ish call of the fused kernel (copy 1) with phase stamps: per-CTA median / max (us from the grid's first CTA)
            st.dbg = torch.zeros(st.grid * 32, dtype=torch.int64, device=dev)
            for _ in range(2):
                st.dbg.zero_()
                (ours_shf if args.trace_arm == 'SF' else ours_sh_ker)(min(1, args.copies - 1), m)
            torch.cuda.synchronize()
            dd = st.dbg.view(st.grid, 32).double().cpu()
            st.dbg = None
            t0 = dd[:, 0].min()
            names = {25: 'init', 26: 'pdl_wait', 27: 'gate', 1: 'tables', 2: 'quant', 28: 'tma1', 8: 'x_ready', 29: 'b1', 9: 'mma1', 10: 'mma_fc1_end', 4: 'fc1_epi_end', 5: 'fc2_end', 6: 't1_end', 7: 'end'}
            line = '  '.join(f'{nm} {float(((dd[:, k] - t0) / 1e3)[dd[:, k] > 0].median()):.1f}/{float(((dd[:, k] - t0) / 1e3)[dd[:, k] > 0].max()):.1f}'
                             for k, nm in names.items() if bool((dd[:, k] > 0).any()))
            allst = gather(line)
            if rank == 0:
                for r_, l_ in enumerate(allst):
                    print(f'   stamps M {m} rank {r_}: {l_}', flush=True)
        if args.trace:   # one cold-ish call (copy 1) with the Xtrace-style phase trace (xtrace.cuh), dumped per rank, shown for rank 0
            import xtrace
            st.dbg = torch.zeros(xtrace.words(st.grid, args.xt_fmt), dtype=torch.int64, device=dev)
            for _ in range(2):
                st.dbg.zero_()
                (ours_shf if args.trace_arm == 'SF' else ours_sh_ker)(min(1, args.copies - 1), m)
            torch.cuda.synchronize()
            tp = args.out / f'xt_m{m}_r{rank}.json'
            xtrace.dump(st.dbg, st.grid, dict(kernel='tpmoe_b2', m=m, rank=rank, label=args.flags, xt_mask=args.xt_mask), tp, fmt=args.xt_fmt)
            st.dbg = None
            dist.barrier()
            if rank == 0:
                import json as _j
                for r_ in range(ndev):
                    xtrace.show(_j.loads((args.out / f'xt_m{m}_r{r_}.json').read_text()))
            dist.barrier()
        if args.rounds == 0:   # correctness only
            allchk = gather(chk)
            if rank == 0:
                print(f'TP{ndev} M {m:4d} check: N vs A {max(c["n_vs_a"] for c in allchk):.5f}  N vs ref {max(c["n_vs_ref"] for c in allchk):.5f}  '
                      f'A vs ref {max(c["a_vs_ref"] for c in allchk):.5f}  N tokmax ref {max(c["n_tokmax_ref"] for c in allchk):.4f}  '
                      f'gate==prod {all(c["gate_eq"] for c in allchk)}  finite {all(c["n_finite"] for c in allchk)}  '
                      f'bitwise N==A {[c["n_eq_a"] for c in allchk]} mismatches {[c["n_ne_a"] for c in allchk]}  '
                  f'S vs A {max(c["s_vs_a"] for c in allchk):.5f} mismatches {[c["s_ne_a"] for c in allchk]}'
                  + (f'  KS2 != KS {[c.get("ks2_ne_ks") for c in allchk]}' if st2 is not None else '')
                  + (f'  SR != S {[c.get("sr_ne_s") for c in allchk]}' if RF is not None else '')
                  + (f'  SF != S {[c.get("sf_ne_s") for c in allchk]} logits {[c.get("sf_lg_ne") for c in allchk]}' if args.arms and 'SF' in args.arms else '')
                  + (f'  SE != S {[c.get("se_ne_s") for c in allchk]} logits {[c.get("se_lg_ne") for c in allchk]}' if args.arms and 'SE' in args.arms else '')
                  + (f'  D != A {[c.get("d_ne_a") for c in allchk]}' if args.arms and ('D' in args.arms or 'DF' in args.arms) else '')
                  + (f'  SD != S {[c.get("sd_ne_s") for c in allchk]} rel {[round(c.get("sd_vs_s", 0), 5) for c in allchk]}' if args.arms and 'SD' in args.arms else ''), flush=True)
            continue
        if args.xt2_time and st2 is not None and st2.dbg is None:   # XTrace OFF/ON pairing: build 2 (TPB2_TRACE=2) records in every timed call
            import xtrace
            st2.dbg = torch.zeros(xtrace.words(st2.grid, 2), dtype=torch.int64, device=dev)
        if args.xt_overhead:   # probes live during the timing (overhead of the Xtrace-style instrumentation vs an unprobed build)
            import xtrace
            st.dbg = torch.zeros(st.grid * (xtrace.HDR + xtrace.WARPS * xtrace.DEPTH * 2), dtype=torch.int64, device=dev)
        graphs = {}
        for lab, fn in arms.items():
            for k in range(ncp):
                fn(k, m)
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            if lab in ('A', 'AR', 'N', 'KN', 'S', 'S2', 'SE', 'SF2', 'D', 'DF', 'KS', 'KS2', 'SR', 'SF', 'SD'):
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
                if ncg is not None:
                    dist.all_reduce(tiny, group=ncg)   # (the host barrier releases ranks ~0.1-1 ms apart: start the timed replays together on the GPUs)
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
                  f'S vs A {max(c["s_vs_a"] for c in allchk):.5f} mismatches {[c["s_ne_a"] for c in allchk]}'
                  + (f'  KS2 != KS {[c.get("ks2_ne_ks") for c in allchk]}' if st2 is not None else '')
                  + (f'  SR != S {[c.get("sr_ne_s") for c in allchk]}' if RF is not None else '')
                  + (f'  SF != S {[c.get("sf_ne_s") for c in allchk]} logits {[c.get("sf_lg_ne") for c in allchk]}' if args.arms and 'SF' in args.arms else '')
                  + (f'  SE != S {[c.get("se_ne_s") for c in allchk]} logits {[c.get("se_lg_ne") for c in allchk]}' if args.arms and 'SE' in args.arms else '')
                  + (f'  D != A {[c.get("d_ne_a") for c in allchk]}' if args.arms and ('D' in args.arms or 'DF' in args.arms) else '')
                  + (f'  SD != S {[c.get("sd_ne_s") for c in allchk]} rel {[round(c.get("sd_vs_s", 0), 5) for c in allchk]}' if args.arms and 'SD' in args.arms else ''), flush=True)
        (args.out / f'rank-{rank}.json').write_text(json.dumps(report, indent=1) + '\n')
        del graphs
        dist.barrier()
    dist.barrier()
    ar.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ndev', type=int, default=4)
    ap.add_argument('--xoff', type=int, default=0, help='first capture-pool row of the M tokens (other inputs for the bitwise checks)')
    ap.add_argument('--sweep', type=int, default=0, help='bitwise sweep over this many consecutive capture offsets per M (no timing)')
    ap.add_argument('--flags3', default=None, help='a third kernel build (arms D / DF)')
    ap.add_argument('--xt-fmt', dest='xt_fmt', type=int, default=1, help='--trace record layout: 1 legacy 16-B ring, 2 XTrace 8-B bounded slabs')
    ap.add_argument('--xt-mask', dest='xt_mask', type=int, default=255, help='XTrace site groups (bits: 0 prologue 1 producer 2 mma 3 b 4 epilogue 5 tail)')
    ap.add_argument('--xt2-time', dest='xt2_time', action='store_true', help='time build 2 (-DTPB2_TRACE=2 in --flags2) with its probes recording: OFF/ON pairs vs build 1')
    ap.add_argument('--rfmax', type=int, default=64, help='in-kernel router (arms SF / SF2 / DF) up to this M (M96/128 need -DTPB2_RRB=1)')
    ap.add_argument('--xbuilds', default=None, help='more builds (4, 5, ..) for --dispatch: "f,f+f,f+.." (+ between builds, , between flags)')
    ap.add_argument('--dispatch', default='', help='arms D / DF: "maxM:build,..." e.g. "8:3,24:1,512:2"')
    ap.add_argument('--model', required=True)
    ap.add_argument('--cap', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layer', type=int, default=20)
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 2, 4, 6, 8, 16, 20, 24, 32, 48, 64, 96, 128])
    ap.add_argument('--copies', type=int, default=4)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--mcbuf', action='store_true', help='all-reduce buffers from torch symmetric memory + NVLS multicast address (TPB2_MC builds)')
    ap.add_argument('--gpu-align', action='store_true', help='tiny NCCL all-reduce right before each timed region (GPU-side rank alignment)')
    ap.add_argument('--arms', nargs='*')
    ap.add_argument('--stamps', action='store_true', help='phase stamps of one fused-kernel call per M (per rank)')
    ap.add_argument('--rfuse', action='store_true', help='arm SR: our router kernel (b2_rfuse.cu) + the fused kernel')
    ap.add_argument('--src2', default='tpmoe_b2.cu', help='kernel source of the KS2 build (e.g. a snapshot)')
    ap.add_argument('--flags2', default=None, help='a second kernel build (comma-separated -D flags), timed as arm KS2 in the same rounds')
    ap.add_argument('--xt-overhead', action='store_true', help='with -DTPB2_TRACE=1 in --flags: keep the xtrace buffer live in the timed calls')
    ap.add_argument('--trace-arm', default='KS', help='--trace: the traced call (KS: precomputed logits; SF: the in-kernel router)')
    ap.add_argument('--trace', action='store_true', help='per-CTA event trace of one fused-kernel call per M (TPB2_TRACE; saved + analyzed)')
    ap.add_argument('--flags', default='', help='extra nvcc -D flags for the fused kernel build (own build dir)')
    args = ap.parse_args()
    assert torch.cuda.device_count() >= args.ndev
    args.out.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(args,), nprocs=args.ndev, join=True)
    print('B2_LAYER_AB_COMPLETE')
