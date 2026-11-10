"""TP8 check + timing of tp8v41 on 8 GPUs, optionally against gc-moe / DeepGEMM MegaMoE EP8 in the same processes.

  python3 v41tp8.py --ts 1 2 4 8 16 32 64 [--arms tp8 gc mega] [--proto a b] [--flags ..] [--check 1]

M = 8 T tokens (T per rank, gc-moe's sweep); our input = the concatenation of the 8 ranks' gc-moe tokens (build_tokens(T, rank)), the EP arms
get their own rank's T tokens (identical token set).
Protocols (per-rank median, then max over ranks; every timed region starts after a cross-rank alignment: tiny NCCL all_reduce + ~100 us
GPU sleep so every rank's replay is enqueued):
  a  gc-moe's: graph of 1 call, 100 replays per window (same inputs: L2 reuse across replays), L2 flushed between windows, 10 + 30 windows
  b  cold L2: graph of 1 call, a 512 MiB L2 flush before every replay, 1 replay per timed region, --cold-iters regions
Correctness (ours): every rank's output bitwise identical, call-to-call identical, rel RMS vs the torch reference of the kernel's numerics
(per-rank partials summed in rank order), cosine vs the bf16 dense reference; also for --skew routing."""
import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v41host as H   # noqa: E402


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


def build_tag(M, extra):
    return f'm{M}_' + ''.join(c for c in '_'.join(extra) if c.isalnum())[:40]


def worker(rank, a):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('nccl', init_method='file://' + a.rdv, world_size=8, rank=rank, device_id=dev)
    gloo = dist.new_group(backend='gloo')
    Ms = [8 * t for t in a.ts] + list(a.ms)
    extra = tuple(a.flags.split()) + (f'-DTPB2_ONESHOT_MAXM={a.oneshot_maxm}',)
    fl = lambda M: H.spec_flags(M, M, 8, extra=extra)
    if 'tp8' in a.arms:   # (builds compiled in parallel: rank r compiles M index r, r + 8, ..)
        t0 = time.time()
        for i, M in enumerate(Ms):
            if i % 8 == rank:
                H.load_ext(tag=build_tag(M, extra), flags=fl(M))
        dist.barrier(group=gloo)
        ext = {M: H.load_ext(tag=build_tag(M, extra), flags=fl(M)) for M in Ms}
        tpv = [('tp8', ext)]
        if a.flags2 is not None:
            extra2 = tuple(a.flags2.split()) + (f'-DTPB2_ONESHOT_MAXM={a.oneshot_maxm2 or a.oneshot_maxm}',)
            fl2 = lambda M: H.spec_flags(M, M, 8, extra=extra2)
            for i, M in enumerate(Ms):
                if i % 8 == rank:
                    H.load_ext(src=a.src2, tag=build_tag(M, extra2), flags=fl2(M))
            dist.barrier(group=gloo)
            tpv.append(('tp8b', {M: H.load_ext(src=a.src2, tag=build_tag(M, extra2), flags=fl2(M)) for M in Ms}))
            if a.swap and rank >= 0:
                tpv = tpv[::-1]
        if a.dbg:
            dfl = lambda M: H.spec_flags(M, M, 8, extra=extra + ('-DTPB2_DBG=1',))
            for i, M in enumerate(Ms):
                if i % 8 == rank:
                    H.load_ext(tag='d' + build_tag(M, extra), flags=dfl(M))
            dist.barrier(group=gloo)
            dext = {M: H.load_ext(tag='d' + build_tag(M, extra), flags=dfl(M)) for M in Ms}
        if rank == 0:
            print(f'builds {time.time() - t0:.0f} s (smem {ext[Ms[0]].SMEM} B)', flush=True)
        E0 = ext[Ms[0]]
        w1, w2 = H.synth_slices(rank, dev)
        raw = H.quantize_slices(w1, w2)
        kl = H.kernel_layout(raw)
        if any(int(getattr(X, 'F2H', 0)) for _, EXT in tpv for X in EXT.values()):   # ((v41 H2) F2H: this rank's H-sharded FC2 weights)
            kl['f2h'] = H.f2h_weights(raw, rank)
        Wv = {vn: H.Weights(next((X for X in EXT.values() if int(getattr(X, 'F2H', 0))), EXT[Ms[0]]), kl) for vn, EXT in tpv}   # ((v41 H2) an F2H build's layout if any)   # (per variant: a build's encode_maps / layout, e.g. W13C)
        if not a.check:
            del w1, w2
        rt = CudaRT(); rt.ck(rt.lib.cudaSetDevice(rank))
        def make_state(EV):   # (per variant: its own workspace + all-reduce buffers -- LLS builds need a sentinel-filled LL region)
            sv = H.RankState(EV, dev, 8, rank)
            ptr = rt.malloc(H.RankState.RS_BYTES(EV))
            if int(getattr(EV, 'LLS', 0)):   # (LLS) every all-reduce slot starts as the sentinel 0xFF.. (readers re-arm their slots)
                rt.ck(rt.lib.cudaMemset(rt.ct.c_void_p(ptr), 0xFF, int(EV.RS_LL_BYTES)))
            hs = [None] * 8
            dist.all_gather_object(hs, rt.handle(ptr), group=gloo)
            sv.link_ptrs([ptr if p == rank else rt.open(h) for p, h in enumerate(hs)])
            if any(int(getattr(X, 'F2H', 0)) for _, EXT2 in tpv for X in EXT2.values()):   # ((v41 H2) F2H) the symmetric gathered-h buffers
                hgb = max(int(getattr(X, 'HG_BYTES', 0)) for _, EXT2 in tpv for X in EXT2.values())   # (counters start at 0; the widest build)
                hp = rt.malloc(hgb)
                hh = [None] * 8
                dist.all_gather_object(hh, rt.handle(hp), group=gloo)
                sv.hgp = [hp if p == rank else rt.open(h) for p, h in enumerate(hh)]
            return sv
        stv = {vn: make_state(EXT[Ms[0]]) for vn, EXT in tpv}
        st = stv[tpv[0][0]]
    if 'gc' in a.arms or 'mega' in a.arms:
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
        Eloc = H.NEXP // 8

        def cast_w(wb):
            L, N, K = wb.shape
            w = torch.empty((L, N, K // 2), device=dev, dtype=torch.int8)
            sf = torch.empty((L, N, K // 32), device=dev, dtype=torch.float)
            for i in range(L):
                w[i], sf[i] = per_token_cast_to_fp4(wb[i], use_ue8m0=True, gran_k=32)
            return w, deep_gemm.transform_sf_into_required_layout(sf, N, K, (1, 32), L)
        g = torch.Generator(device=dev).manual_seed(7 + rank)
        ew1 = torch.randn(Eloc, 2 * H.INTER_FULL, H.DIM, dtype=torch.bfloat16, device=dev, generator=g) * 0.05
        ew2 = torch.randn(Eloc, H.DIM, H.INTER_FULL, dtype=torch.bfloat16, device=dev, generator=g) * 0.05
        l1, l2 = cast_w(ew1), cast_w(ew2)
        del ew1, ew2
        ops = {}
        maxT = max(a.ts) if a.ts else 1
        if 'mega' in a.arms:
            ops['mega'] = (deep_gemm.fp8_fp4_mega_moe, deep_gemm.transform_weights_for_mega_moe(l1, l2, activation='swiglu'),
                           deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, H.NEXP, maxT, H.TOPK, H.DIM, H.INTER_FULL, activation='swiglu'))
        if 'gc' in a.arms:
            import gc_moe
            ops['gc'] = (gc_moe.fp8_fp4_gc_moe, gc_moe.transform_weights_for_gc_moe(l1, l2, activation='swiglu'),
                         gc_moe.get_symm_buffer_for_gc_moe(dist.group.WORLD, H.NEXP, maxT, H.TOPK, H.DIM, H.INTER_FULL, activation='swiglu'))
    trt_arms = [n for n in ('trt', 'trtn') if n in a.arms]
    if trt_arms:
        import trtarm
        if 'tp8' not in a.arms:
            w1, w2 = H.synth_slices(rank, dev)
            raw = H.quantize_slices(w1, w2)
        trtW = trtarm.prep_weights(raw, ipad=a.trt_ipad)
        trt_moe = trtarm.TrtMoE(trtW, dev)
        trt_ar = trtarm.TrtAR(rank, dist.group.WORLD) if 'trt' in trt_arms else None
        trt_tmp = torch.empty(512, H.DIM, dtype=torch.bfloat16, device=dev)
        if rank == 0:
            print(f'trt arm: flashinfer trtllm-gen routed MoE, intermediate padded {H.INTER} -> {a.trt_ipad}; all-reduce: '
                  f'trt = flashinfer allreduce_fusion(kAllReduce, backend trtllm, auto one-shot / two-shot), trtn = NCCL all_reduce', flush=True)
    torch.cuda.synchronize(); dist.barrier(group=gloo)
    tiny = torch.zeros(1, device=dev)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)

    def align():
        dist.all_reduce(tiny)
        torch.cuda._sleep(a.sleep_cycles)

    def time_graph(g, proto):
        res = []
        if proto == 'a':
            for wi in range(10 + a.windows):
                dist.barrier(group=gloo)
                flush.zero_()
                align()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(a.inner):
                    g.replay()
                e1.record(); torch.cuda.synchronize()
                if wi >= 10:
                    res.append(e0.elapsed_time(e1) * 1e3 / a.inner)
        else:
            for wi in range(3 + a.cold_iters):
                if wi % 25 == 0:
                    dist.barrier(group=gloo)
                flush.zero_()
                align()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record()
                torch.cuda.synchronize()
                if wi >= 3:
                    res.append(e0.elapsed_time(e1) * 1e3)
        med = statistics.median(res)
        rs = sorted(res)
        k = len(rs) // 10
        tm = statistics.mean(rs[k:len(rs) - k])   # (trimmed mean: the event timer is quantized at ~2 us on B300, medians snap to it)
        allm = [None] * 8
        dist.all_gather_object(allm, (med, tm), group=gloo)
        return max(v[0] for v in allm), [v[0] for v in allm], max(v[1] for v in allm)

    def capture(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); dist.barrier(group=gloo)
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize(); dist.barrier(group=gloo)
        return g

    def b2b_run(vn, EXT, plan, rounds, ncalls=24, mixed=False):
        """rank-skewed, input-cycling (optionally mixed-M) back-to-back eager bursts.  plan = [(M, (x, ids, rw))]; refs = isolated single
        calls (barrier + sync around each).  Per round: ncalls calls (single: cycling the plan; mixed: drawn at random, the SAME sequence on
        every rank), each into its own NaN-poisoned output; sleeps: before the burst rank (r + round) % 8 x 16000 cycles (the late rank
        rotates), before call i rank i % 8 sleeps 8000 cycles (B300b's pattern) and every rank ~1/4 of the calls 500..8000 cycles (per-rank
        rng); every output compared bitwise (on the GPU) with its case's isolated output.  Returns per-case bad / calls summed over ranks."""
        import random
        W, st_ = Wv[vn], stv[vn]
        refs = []
        for M, (cx, cids, crw) in plan:
            o = torch.empty(M, H.DIM, dtype=torch.bfloat16, device=dev)
            torch.cuda.synchronize(); dist.barrier(group=gloo)
            H.run(st_, W, cx, cids, crw, o, E=EXT[M])
            torch.cuda.synchronize()
            refs.append(o)
        same = True
        for o in refs:
            allo = [torch.empty_like(o) for _ in range(8)]
            dist.all_gather(allo, o)
            same = same and all(torch.equal(allo[0], v) for v in allo)
        Mx = max(c[0] for c in plan)
        outs = [torch.empty(Mx, H.DIM, dtype=torch.bfloat16, device=dev) for _ in range(ncalls)]
        nc = len(plan)
        bad = torch.zeros(nc, dtype=torch.int64, device=dev)
        calls = [0] * nc
        rng_s, rng_r = random.Random(12345), random.Random(7919 * rank + 17)
        torch.cuda.synchronize(); dist.barrier(group=gloo)
        t0 = time.time()
        for r in range(rounds):
            seq = [rng_s.randrange(nc) for _ in range(ncalls)] if mixed else [i % nc for i in range(ncalls)]
            for o in outs:
                o.fill_(float('nan'))
            torch.cuda._sleep(16000 * ((rank + r) % 8))
            for i, ci in enumerate(seq):
                if i % 8 == rank:
                    torch.cuda._sleep(8000)
                if rng_r.random() < 0.25:
                    torch.cuda._sleep(rng_r.randint(500, 8000))
                M, (cx, cids, crw) = plan[ci]
                H.run(st_, W, cx, cids, crw, outs[i][:M], E=EXT[M])
                calls[ci] += 1
            for i, ci in enumerate(seq):
                bad[ci] += (outs[i][:plan[ci][0]] != refs[ci]).any().to(torch.int64)
            if r % 16 == 15:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        dt = time.time() - t0
        allr = [None] * 8
        dist.all_gather_object(allr, bad.cpu().tolist(), group=gloo)
        nb = [sum(v[i] for v in allr) for i in range(nc)]
        return same, nb, [8 * c for c in calls], dt

    def run_trt(nm, M, T, row):
        gb = H.route_batch(T if T else (M + 7) // 8, dev, a.routing, a.zipf_s)
        x, ids, rw = (v[:M].contiguous() for v in gb)
        moe_out = trt_tmp[:M]
        out = torch.empty(M, H.DIM, dtype=torch.bfloat16, device=dev)
        if nm == 'trt':
            fn = lambda: trt_ar(trt_moe(x, ids, rw, moe_out), out)
        else:
            def fn():
                out.copy_(trt_moe(x, ids, rw, moe_out))
                dist.all_reduce(out)
        with trtarm.autotune_ctx():   # (SGLang-style startup autotune of the trtllm-gen MoE tactics at this M)
            trt_moe(x, ids, rw, moe_out)
        torch.cuda.synchronize(); dist.barrier(group=gloo)
        m = ''
        if a.check:
            fn(); torch.cuda.synchronize(); o1 = out.clone(); fn(); torch.cuda.synchronize()
            det = bool(torch.equal(o1, out))
            allo = [torch.empty_like(o1) for _ in range(8)]
            dist.all_gather(allo, o1)
            same = all(torch.equal(allo[0], v) for v in allo)
            part = H.ref_rank(raw, x, ids, rw)
            allp = [torch.empty_like(part) for _ in range(8)]
            dist.all_gather(allp, part)
            ref = H.combine_tp(allp)
            dref = H.ref_dense_rank(w1, w2, x, ids, rw)
            dist.all_reduce(dref)
            dref = H.bf16r(dref)
            r1, c1, c0 = H.stats(o1, ref), H.cosine(o1, dref), H.cosine(ref, dref)
            allr = [None] * 8
            dist.all_gather_object(allr, (r1, c1, det), group=gloo)
            m = (f' | {nm}:{a.routing[:4]} rel {max(v[0] for v in allr):.5f} cos {min(v[1] for v in allr):.5f} (ref {c0:.5f}) '
                 f'ranks_same {same} det {all(v[2] for v in allr)}')
        g = capture(fn)
        for p in a.proto:
            _t = time_graph(g, p); row[(nm, p)] = _t[0]; row[(nm, p + 't')] = _t[2]
        del g
        return m

    cases = [tuple(int(v) for v in c.split(':')) for c in a.dec_cases.split(',')] if a.routing == 'dec' else [None]
    for T, M, case in [(t, 8 * t, c) for t in a.ts for c in cases] + [(None, m, c) for m in a.ms for c in cases]:
        if case is not None:
            H.DEC_CASE = case
        row = {}
        msg = ''
        if rank == 0:
            gb = H.route_batch(T if T else (M + 7) // 8, dev, a.routing, a.zipf_s)
            rr, emax, U = H.ep_stats(gb[1][:M])
            msg += f' | routing {a.routing}' + (f' s={a.zipf_s}' if a.routing == 'zipf' else '') + f' EPrecv max/mean {rr:.2f} U {U} maxload {emax}'
        for nm in [n for n in ('mega', 'gc') if n in a.arms and T and a.order == 'rev']:
            op, wts, buf = ops[nm]
            x_bf, idx, w = H.route_tokens(T, rank, dev, a.routing, a.zipf_s)
            x8, xsf = per_token_cast_to_fp8(x_bf, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            buf.x[:T].copy_(x8); buf.x_sf[:T].copy_(xsf); buf.topk_idx[:T].copy_(idx.to(torch.int64)); buf.topk_weights[:T].copy_(w)
            y = torch.empty(T, H.DIM, dtype=torch.bfloat16, device=dev)
            torch.cuda.synchronize(); dist.barrier(group=gloo)
            fn = lambda: op(y, *wts, buf, recipe=(1, 1, 32), activation='swiglu', activation_clamp=10.0, activation_alpha=1.0, fast_math=True)
            g = capture(fn)
            for p in a.proto:
                _t = time_graph(g, p); row[(nm, p)] = _t[0]; row[(nm, p + 't')] = _t[2]
            del g
        if a.order == 'rev':
            for nm in [n for n in ('trtn', 'trt') if n in a.arms]:
                msg += run_trt(nm, M, T, row)
        for vn, EXT in (tpv if 'tp8' in a.arms else []):
                E = EXT[M]
                W = Wv[vn]
                st = stv[vn]
                gb = H.route_batch(T if T else (M + 7) // 8, dev, a.routing, a.zipf_s)   # (M = 8 T, or the --ms M: ceil(M / 8) tokens per rank, truncated)
                cases = [(a.routing[:4], tuple(v[:M].contiguous() for v in gb))] + ([('skew', H.skew_batch(M, dev, seed=M, sigma=a.skew_sigma))] if a.skew else [])
                x, ids, rw = cases[0][1]
                out = torch.empty(M, H.DIM, dtype=torch.bfloat16, device=dev)
                if a.check:
                    for nm, (cx, cids, crw) in cases:
                        o1 = torch.empty(M, H.DIM, dtype=torch.bfloat16, device=dev); o2 = torch.empty_like(o1)
                        for _ in range(3):
                            H.run(st, W, cx, cids, crw, o1, E=E)
                        H.run(st, W, cx, cids, crw, o2, E=E)
                        torch.cuda.synchronize()
                        det = bool(torch.equal(o1, o2))
                        allo = [torch.empty_like(o1) for _ in range(8)]
                        dist.all_gather(allo, o1)
                        same = all(torch.equal(allo[0], v) for v in allo)
                        part = (H.ref_rank_h(raw, W.f2h_w2, W.f2h_s2, cx, cids, crw, rank) if int(getattr(E, 'F2H', 0))
                                else H.ref_rank(raw, cx, cids, crw))
                        allp = [torch.empty_like(part) for _ in range(8)]
                        dist.all_gather(allp, part)
                        ref = H.combine_tp(allp)
                        dref = H.ref_dense_rank(w1, w2, cx, cids, crw)
                        dist.all_reduce(dref)
                        dref = H.bf16r(dref)
                        r1, c1, c0 = H.stats(o1, ref), H.cosine(o1, dref), H.cosine(ref, dref)
                        allr = [None] * 8
                        dist.all_gather_object(allr, (r1, c1, det), group=gloo)
                        U = len(torch.unique(cids))
                        msg += (f' | {vn}:{nm} U={U} rel {max(v[0] for v in allr):.5f} cos {min(v[1] for v in allr):.5f} (ref {c0:.5f}) '
                                f'ranks_same {same} det {all(v[2] for v in allr)}')
                        if a.stress > 0:   # back-to-back stress: a.stress rounds x 8 eager calls (no flush / gate), every output == o1 bitwise
                            outs = [torch.empty_like(o1) for _ in range(8)]
                            mism = torch.zeros(1, dtype=torch.int64, device=dev)
                            bad = torch.zeros(1, dtype=torch.int64, device=dev)
                            torch.cuda.synchronize(); dist.barrier(group=gloo)
                            t0s = time.time()
                            for rr in range(a.stress):
                                for i in range(8):
                                    H.run(st, W, cx, cids, crw, outs[i], E=E)
                                for i in range(8):
                                    ne = (outs[i] != o1).sum()
                                    mism += ne
                                    bad += (ne > 0).to(torch.int64)
                                if rr % 64 == 63:
                                    torch.cuda.synchronize()
                            torch.cuda.synchronize()
                            sres = [None] * 8
                            dist.all_gather_object(sres, (int(mism.item()), int(bad.item())), group=gloo)
                            msg += (f' | STRESS {vn}:{nm} {a.stress} x 8 calls per rank ({8 * a.stress * 8} total): bad calls {sum(v[1] for v in sres)} '
                                    f'mismatched elements {sum(v[0] for v in sres)} ({time.time() - t0s:.0f} s)')
                if a.b2b > 0:   # rank-skewed input-cycling bursts at this M: the routing's batch + 2 skewed batches (other x)
                    s1 = H.skew_batch(M, dev, seed=1000 + M, sigma=a.skew_sigma)
                    s2 = H.skew_batch(M, dev, seed=2000 + M, sigma=a.skew_sigma)
                    plan = [(M, (x, ids, rw)), (M, s1), (M, (s2[0] * 0.75, s2[1], s2[2]))]
                    same_, nb, nca, dt = b2b_run(vn, EXT, plan, a.b2b)
                    msg += (f' | B2B {vn} F2H={int(getattr(E, "F2H", 0))} NMAX={int(getattr(E, "NMAX", 0)) or "?"} refs ranks_same {same_} bad {sum(nb)}/{sum(nca)} '
                            f'(per input {nb}) ({dt:.0f} s)')
                if a.b2b_only:
                    continue
                g = capture(lambda: H.run(st, W, x, ids, rw, out, E=E))
                for p in a.proto:
                    _t = time_graph(g, p); row[(vn, p)] = _t[0]; row[(vn, p + 't')] = _t[2]
                del g
                if a.dbg:   # stamps of one cold aligned launch (us from the earliest start over all ranks; globaltimer differs per GPU only slightly)
                    G = st.grid
                    dbg = torch.zeros(G * 32, dtype=torch.int64, device=dev)
                    for it in range(3):
                        dbg.zero_(); flush.zero_()
                        dist.barrier(group=gloo); torch.cuda.synchronize()
                        align()
                        H.run(st, W, x, ids, rw, out, dbg=dbg, E=dext[M])
                        torch.cuda.synchronize()
                    d = dbg.view(G, 32).cpu().double()
                    t0 = d[:, 0].min()
                    f = lambda k: ((d[:, k][d[:, k] > 0] - t0) / 1e3)
                    keys = (26, 1, 11, 3, 4, 2, 9, 25, 27, 5, 12, 13, 6, 21, 22, 10, 28, 7, 17, 23, 24)   # (F2H: 10 tail entry, 21 T1 tile ready, 28 T1 sends done)   # (F2H: 2 = sender done, 9 = first FC2-H panel ready)
                    row_ = {k: (float(f(k).median()), float(f(k).max())) for k in keys if (d[:, k] > 0).any()}
                    for k in ((12, 13, 6) if int(getattr(E, 'F2H', 0)) else ()) + (22, 17, 23, 24):   # (cycle sums -> us at 1.9 GHz: F2H 12 / 13 / 6 sender + B warp;
                        # 22 MMA FC1 (+FC2-H) data waits, 17 MMA acc-empty waits, 23 MMA FC1 issue, 24 producer FC1 stage waits)   # (12 sender write waits, 13 sender issue + reads + signal, 6 B-warp counter waits, 21 producer ready-list waits, 22 MMA FC1 (+ FC2-H) data waits)
                            row_[k] = (float(d[:, k].median()) / 1900.0, float(d[:, k].max()) / 1900.0)
                    allr = [None] * 8
                    dist.all_gather_object(allr, (float(t0), row_), group=gloo)
                    if rank == 0:
                        g0 = min(r[0] for r in allr)
                        print(f'  DBG M={M}: med/max us from each rank start (26 pdl 1 tables 11 1stFC1epi 3 fc1pushed 4 lastH 25 1stFC2 27 mmaExit '
                              f'5 roles 12 T1rdy 13 T1item 6 T1done 7 end)', flush=True)
                        for r, (s0, rw_) in enumerate(allr):
                            print(f'   r{r} start {(s0 - g0) / 1e3:7.2f} | ' + ' '.join(f'{k}:{v[0]:.1f}/{v[1]:.1f}' for k, v in rw_.items()), flush=True)
        if a.order != 'rev':
            for nm in [n for n in ('trt', 'trtn') if n in a.arms]:
                msg += run_trt(nm, M, T, row)
        for nm in [n for n in ('gc', 'mega') if n in a.arms and T and a.order != 'rev']:
            op, wts, buf = ops[nm]
            x_bf, idx, w = H.route_tokens(T, rank, dev, a.routing, a.zipf_s)
            x8, xsf = per_token_cast_to_fp8(x_bf, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            buf.x[:T].copy_(x8); buf.x_sf[:T].copy_(xsf); buf.topk_idx[:T].copy_(idx.to(torch.int64)); buf.topk_weights[:T].copy_(w)
            y = torch.empty(T, H.DIM, dtype=torch.bfloat16, device=dev)
            torch.cuda.synchronize(); dist.barrier(group=gloo)
            fn = lambda: op(y, *wts, buf, recipe=(1, 1, 32), activation='swiglu', activation_clamp=10.0, activation_alpha=1.0, fast_math=True)
            g = capture(fn)
            for p in a.proto:
                _t = time_graph(g, p); row[(nm, p)] = _t[0]; row[(nm, p + 't')] = _t[2]
            del g
        if rank == 0:
            line = f'RES T={T if T else 0:3d} M={M:4d}' + (f' case L{case[0]}s{case[1]}' if case is not None else '')
            for (nm, p), v in row.items():
                line += f' | {nm}.{p} {v:7.1f}'
            print(line + msg, flush=True)
    if a.b2b_mix > 0 and 'tp8' in a.arms:   # mixed-M bursts over every M build of the arm (shared workspace / epochs / LL + gathered-h buffers)
        for vn, EXT in tpv:
            plan = []
            for M in sorted(set(Ms)):
                gb = H.route_batch((M + 7) // 8, dev, a.routing, a.zipf_s)
                s1 = H.skew_batch(M, dev, seed=1000 + M, sigma=a.skew_sigma)
                s2 = H.skew_batch(M, dev, seed=2000 + M, sigma=a.skew_sigma)
                plan += [(M, tuple(v[:M].contiguous() for v in gb)), (M, s1), (M, (s2[0] * 0.75, s2[1], s2[2]))]
            same_, nb, nca, dt = b2b_run(vn, EXT, plan, a.b2b_mix, mixed=True)
            if rank == 0:
                per = {}
                for (M, _), b_, c_ in zip(plan, nb, nca):
                    v = per.setdefault(M, [0, 0]); v[0] += b_; v[1] += c_
                print(f'B2B-MIX {vn}: refs ranks_same {same_} bad {sum(nb)}/{sum(nca)} rank-calls ({dt:.0f} s); per M (F2H) bad/calls: '
                      + ' '.join(f'M{M}({int(getattr(EXT[M], "F2H", 0))}) {v[0]}/{v[1]}' for M, v in per.items()), flush=True)
    torch.cuda.synchronize(); dist.barrier(group=gloo)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts', type=int, nargs='*', default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument('--ms', type=int, nargs='*', default=[])   # (extra M for our arm only, e.g. 1 2 4)
    ap.add_argument('--arms', nargs='+', default=['tp8'])
    ap.add_argument('--proto', nargs='+', default=['a', 'b'])
    ap.add_argument('--flags', default='')
    ap.add_argument('--src2', default='tp8v41.cu')   # (source of the tp8b build: e.g. an older commit's generated tp8v41.cu)
    ap.add_argument('--flags2', default=None)   # (a second build of ours timed in the same run: arm tp8b)
    ap.add_argument('--oneshot-maxm2', type=int, default=0)
    ap.add_argument('--swap', type=int, default=0)   # (tp8b first)
    ap.add_argument('--oneshot-maxm', type=int, default=128)
    ap.add_argument('--check', type=int, default=1)
    ap.add_argument('--stress', type=int, default=0)   # (with --check 1: rounds of 8 back-to-back eager calls compared bitwise to the first call)
    ap.add_argument('--skew', type=int, default=1); ap.add_argument('--skew-sigma', type=float, default=1.0)
    ap.add_argument('--windows', type=int, default=30); ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--cold-iters', type=int, default=50)
    ap.add_argument('--sleep-cycles', type=int, default=200000)
    ap.add_argument('--dbg', type=int, default=0)
    ap.add_argument('--b2b', type=int, default=0)   # (with --check 1: rounds of 24 rank-skewed input-cycling eager calls per M, bitwise vs isolated calls)
    ap.add_argument('--b2b-mix', type=int, default=0)   # (rounds of 24 rank-skewed calls drawn over all M builds, bitwise vs isolated calls)
    ap.add_argument('--b2b-only', type=int, default=0)   # (skip the timing)
    ap.add_argument('--order', default='fwd')   # (fwd: ours, trt, trtn, gc, mega per M; rev: the reverse)
    ap.add_argument('--routing', default='uniform', choices=['uniform', 'zipf', 'real', 'dec'])
    ap.add_argument('--dec-cases', default='10:32,20:32,30:32,10:96,20:96,30:96')   # (dec: layer:step cases per M)
    ap.add_argument('--zipf-s', type=float, default=1.2)
    ap.add_argument('--trt-ipad', type=int, default=384)   # (trtllm-gen: 2 I must be a multiple of 128 and I = 320 has no valid config; 384 is the smallest that runs)
    a = ap.parse_args()
    a.rdv = tempfile.mktemp(prefix='v41_rdv_', dir=os.environ.get('TP8_BUILD', '/tmp'))
    torch.multiprocessing.spawn(worker, args=(a,), nprocs=8, join=True)


if __name__ == '__main__':
    main()
