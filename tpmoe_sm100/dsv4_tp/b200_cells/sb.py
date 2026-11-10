"""suite_b200 harness: one cell (model x TP size) on N B200 GPUs, arms timed in one process, per M.

  python sb.py --model dsv4f --tp 4 --ms 8 16 32 64 128 256 384 512 --arms ours gc mega --routing real --proto b a --check 1
  python sb.py --model dsv4f --tp 4 ... --arms trt          (trtllm-gen in its own process, as on B300)

Arms (all with the same total tokens M per call):
  ours  our TP kernel of the cell (DSV4-Flash TP4: v77 routed-only; DSV4.1-Flash TP8: tp8v41), x bf16 in, ids / weights given,
        all-reduced bf16 out on every rank (in-kernel AR)
  trt   flashinfer trtllm-gen MXFP4 x MXFP8 routed MoE on the TP slice + mxfp8_quantize in graph + flashinfer TRT-LLM IPC all-reduce
  gc    gc-moe EP (CuTe DSL + green contexts), T = M / N tokens per rank: input quant (one Triton kernel, in graph) + dispatch +
        experts + combine
  mega  DeepGEMM fp8_fp4_mega_moe EP (same buffers / quant as gc)
Routing: real = the first M rows of the model's layer-20 serving trace (EP rank r = rows r T ..); uniform = gc-moe style random.
Protocols: b (headline) = cold L2 per call (512 MiB flush + rank alignment + sleep before every replay); a = gc-moe's 100-replay windows.
Correctness (ours / trt, --check 1): ranks bitwise identical, call-to-call deterministic, rel vs the numerics reference (per-rank
torch emulation summed in rank order), cos vs the bf16 dense reference; v77: mismatches vs trtllm-gen partials summed in rank order."""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sbcommon as C   # noqa: E402

OURS = {('dsv4f', 4): 'v77', ('dsv4f', 8): 'v77', ('dsv41', 8): 'tp8v41', ('dsv41', 4): 'tp8v41'}


def worker(rank, a):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    N = a.tp
    dist.init_process_group('nccl', init_method='file://' + a.rdv, world_size=N, rank=rank, device_id=dev)
    gloo = dist.new_group(backend='gloo')
    sh = C.Shape(a.model, N)
    Ms = a.ms
    log = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)
    t0 = time.time()
    rt = C.CudaRT(); rt.ck(rt.lib.cudaSetDevice(rank))
    ipc = lambda n: C.ipc_table(rt, n, rank, N, gloo)
    need_tp = any(n in a.arms for n in ('ours', 'trt')) or a.check
    w1 = w2 = raw = trtW = None
    if need_tp:
        w1, w2 = C.synth_slices(sh, rank, dev)
        raw = C.quantize_slices(sh, w1, w2)
    arms = {}
    kind = a.ours or OURS.get((a.model, N))
    kind2 = a.ours2_kind or kind
    if 'ours' in a.arms:
        if kind == 'v77':
            import sbtrt
            import arm_v77
            trtW = sbtrt.prep_weights(sh, raw)
            arms['ours'] = arm_v77.Arm(sh, rank, dev, gloo, ipc, trtW, Ms, extra=tuple(a.flags.split()))
        elif kind == 'tp8v41':
            import arm_tp8v41
            arms['ours'] = arm_tp8v41.Arm(sh, rank, dev, gloo, ipc, raw, Ms, extra=tuple(a.flags.split()), oneshot_maxm=a.oneshot_maxm)
        elif kind == 'disp':   # (DSV4-Flash dispatch: v77 below SB_V4FT_MINM (384), v4ft from it; separate workspaces)
            import sbtrt
            import arm_disp
            import arm_v4ft
            trtW = sbtrt.prep_weights(sh, raw)
            arms['ours'] = arm_disp.Arm(sh, rank, dev, gloo, ipc, raw, trtW, Ms, extra=tuple(a.flags.split()), oneshot_maxm=a.oneshot_maxm,
                                        per_m=arm_v4ft.parse_dispatch(a.dispatch) if a.dispatch else None)
        elif kind == 'v4ft':   # (DSV4-Flash, tp8v41 lineage: suite_b300b's port_v4f.py over tp8_v41_b300 e066ec9)
            import arm_v4ft
            arms['ours'] = arm_v4ft.Arm(sh, rank, dev, gloo, ipc, raw, Ms, extra=tuple(a.flags.split()), oneshot_maxm=a.oneshot_maxm,
                                        per_m=arm_v4ft.parse_dispatch(a.dispatch))
        else:
            raise SystemExit(f'no ours arm for {a.model} TP{N}')
        if a.flags2 is not None:
            if kind2 == 'v4ft':   # (--ours2-kind v4ft: cross-kind same-run A/B, per-M flags from --dispatch2)
                import arm_v4ft
                arms['ours2'] = arm_v4ft.Arm(sh, rank, dev, gloo, ipc, raw, Ms, extra=tuple(a.flags2.split()),
                                             oneshot_maxm=a.oneshot_maxm2 or a.oneshot_maxm, per_m=arm_v4ft.parse_dispatch(a.dispatch2))
            elif kind2 == 'v77':
                if trtW is None:
                    import sbtrt
                    import arm_v77
                    trtW = sbtrt.prep_weights(sh, raw)
                arms['ours2'] = arm_v77.Arm(sh, rank, dev, gloo, ipc, trtW, Ms, extra=tuple(a.flags2.split()))
            else:
                arms['ours2'] = arm_tp8v41.Arm(sh, rank, dev, gloo, ipc, raw, Ms, extra=tuple(a.flags2.split()), oneshot_maxm=a.oneshot_maxm2 or a.oneshot_maxm, src=a.src2)
        if a.dbg and kind == 'tp8v41':
            arms['dbg'] = arm_tp8v41.Arm(sh, rank, dev, gloo, ipc, raw, Ms, extra=tuple(a.flags.split()) + ('-DTPB2_DBG=1',), oneshot_maxm=a.oneshot_maxm)
        elif a.dbg and kind == 'v4ft':
            dfl = ('-DTPB2_TRACE=2', f'-DTPB2_XT_MASK={a.xt_mask}') if a.trace else ('-DTPB2_DBG=1',)
            arms['dbg'] = arm_v4ft.Arm(sh, rank, dev, gloo, ipc, raw, Ms, extra=tuple(a.flags.split()) + dfl, oneshot_maxm=a.oneshot_maxm,
                                       per_m=arm_v4ft.parse_dispatch(a.dispatch))
        elif a.dbg and kind == 'v77':
            dfl = ('-DTPB2_TRACE=2', f'-DTPB2_XT_MASK={a.xt_mask}') if a.trace else ('-DTPB2_DBG=1',)
            arms['dbg'] = arm_v77.Arm(sh, rank, dev, gloo, ipc, trtW, Ms, extra=tuple(a.flags.split()) + dfl)
        log(f'ours = {kind}: builds ready {time.time() - t0:.0f} s' + (f' (ours2 flags: {a.flags2})' if a.flags2 is not None else ''))
    trt = None
    if 'trt' in a.arms or (a.check and kind == 'v77' and 'ours' in a.arms and a.trtgen_cmp):
        import sbtrt
        ipad = a.trt_ipad or sh.INTER
        trtWp = trtW if (trtW is not None and ipad == sh.INTER) else sbtrt.prep_weights(sh, raw, ipad=ipad)
        trt = (sbtrt.TrtMoE(sh, trtWp, dev), sbtrt.TrtAR(sh, rank, dist.group.WORLD) if 'trt' in a.arms else None)
        log(f'trt: flashinfer trtllm-gen routed MoE (intermediate {sh.INTER}' + (f' padded to {ipad}' if ipad != sh.INTER else '') +
            ') + flashinfer TRT-LLM IPC all-reduce')
    ep = {}
    if 'gc' in a.arms or 'mega' in a.arms:
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp4
        Eloc = sh.NEXP // N

        def cast_w(wb):
            L, Nn, K = wb.shape
            w = torch.empty((L, Nn, K // 2), device=dev, dtype=torch.int8)
            sf = torch.empty((L, Nn, K // 32), device=dev, dtype=torch.float)
            for i in range(L):
                w[i], sf[i] = per_token_cast_to_fp4(wb[i], use_ue8m0=True, gran_k=32)
            return w, deep_gemm.transform_sf_into_required_layout(sf, Nn, K, (1, 32), L)
        g = torch.Generator(device=dev).manual_seed(7 + rank)
        l1 = cast_w(torch.randn(Eloc, 2 * sh.INTER_FULL, sh.DIM, dtype=torch.bfloat16, device=dev, generator=g) * 0.05)
        l2 = cast_w(torch.randn(Eloc, sh.DIM, sh.INTER_FULL, dtype=torch.bfloat16, device=dev, generator=g) * 0.05)
        maxT = max(1, max(Ms) // N)
        if 'mega' in a.arms:
            ep['mega'] = (deep_gemm.fp8_fp4_mega_moe, deep_gemm.transform_weights_for_mega_moe(l1, l2, activation='swiglu'),
                          deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, sh.NEXP, maxT, sh.TOPK, sh.DIM, sh.INTER_FULL, activation='swiglu'))
        if 'gc' in a.arms:
            import gc_moe
            ep['gc'] = (gc_moe.fp8_fp4_gc_moe, gc_moe.transform_weights_for_gc_moe(l1, l2, activation='swiglu'),
                        gc_moe.get_symm_buffer_for_gc_moe(dist.group.WORLD, sh.NEXP, maxT, sh.TOPK, sh.DIM, sh.INTER_FULL, activation='swiglu'))
        del l1, l2
        epq = C.ep_quant_kernel()
        log(f'EP arms {list(ep)}: {Eloc} experts / rank, buffers for T <= {maxT}')
    torch.cuda.synchronize(); dist.barrier(group=gloo)
    tm = C.Timer(dev, gloo, N, cold_iters=a.cold_iters, windows=a.windows, inner=a.inner, sleep_cycles=a.sleep_cycles)

    def gather_t(t):
        allt = [torch.empty_like(t) for _ in range(N)]
        dist.all_gather(allt, t)
        return allt

    def gather_o(o):
        allo = [None] * N
        dist.all_gather_object(allo, o, group=gloo)
        return allo

    def akind(nm, M):   # (the kernel family serving arm nm at M: dispatch arms per M)
        A_ = arms.get(nm)
        if hasattr(A_, 'kind_of'):
            return A_.kind_of(M)
        return getattr(A_, 'kind', kind if nm in ('ours', 'dbg') else kind2)

    def check_tp(nm, fn, out, x, ids, rw, bf16w):
        fn(); torch.cuda.synchronize(); o1 = out.clone()
        for _ in range(0 if int(os.environ.get('SB_CHECK_NODET', '0')) else 2):   # (test hook: no extra eager calls)
            fn()
        torch.cuda.synchronize()
        det = bool(torch.equal(o1, out))
        allo = gather_t(o1)
        same = all(torch.equal(allo[0], v) for v in allo)
        A_ = arms.get(nm)
        ak = akind(nm, int(x.shape[0]))
        if ak == 'v4ft':   # (v4ft: its own exact numerics reference; collective for F2H builds)
            ref = C.combine_tp(gather_t(A_.ref_partial(x, ids, rw).float()))
        elif kind == 'tp8v41' and getattr(A_, 'f2h', False) and int(getattr(A_.ext.get(int(x.shape[0])), 'F2H', 0)):   # ((v41 H2) its own numerics)
            ref = C.combine_tp(gather_t(A_.H.ref_rank_h(raw, A_.W.f2h_w2, A_.W.f2h_s2, x, ids, rw, rank).float()))
        else:
            ref = C.combine_tp(gather_t(C.ref_rank(sh, raw, x, ids, rw, bf16_weights=bf16w)))
        dref = C.ref_dense_rank(sh, w1, w2, x, ids, rw)
        dist.all_reduce(dref)
        r1, c1, c0 = C.rel(o1, ref), C.cosine(o1, C.bf16r(dref)), C.cosine(ref, C.bf16r(dref))
        nbad = int((o1.view(torch.int16) != ref.to(torch.bfloat16).view(torch.int16)).sum())
        extra = f' mism_vs_emu {max(gather_o(nbad))}/{o1.numel()}'
        if nm in ('ours', 'ours2') and ak == 'v77' and trt is not None and a.trtgen_cmp:   # bitwise vs trtllm-gen (production routed numerics), rank-order sum
            p = torch.empty_like(o1)
            trt[0](x, ids, rw, p); torch.cuda.synchronize()
            prod = C.combine_tp(gather_t(p)).to(torch.bfloat16)
            extra = f' mism_vs_trtgen {int((o1.view(torch.int16) != prod.view(torch.int16)).sum())}'
        allr = gather_o((r1, c1, det))
        return (f' | {nm}: rel {max(v[0] for v in allr):.5f} cos {min(v[1] for v in allr):.5f} (ref {c0:.5f}) ranks_same {same} '
                f'det {all(v[2] for v in allr)}' + extra)

    CLK = int(os.environ.get('SB_CLK', '0'))
    PROBE = int(os.environ.get('SB_PROBE', '0'))

    def probe_all():   # (SB_PROBE) SM availability probe on every rank: the worst rank's line
        import smprobe
        if smprobe._EXT is None:   # (rank 0 builds, the others load)
            if rank == 0:
                smprobe._ext()
            dist.barrier(group=gloo)
            smprobe._ext()
        d = smprobe.probe(dev)
        allp = gather_o(d)
        w = max(range(len(allp)), key=lambda r_: (allp[r_]['cyc_ratio'], allp[r_]['start_max']))
        return f'r{w} ' + smprobe.fmt(allp[w])


    def b2b_test(nm, M):
        """--b2b K: back-to-back correctness. K calls on the routing cases (cycled; own input / output tensors each), references from
        sync- and barrier-separated calls; then (1) eager bursts of the K calls with no host sync, rank-skewed starts, (2) bursts where
        rank (rep % N) sleeps between its calls (ranks drift apart), (3) one CUDA graph of the K calls, replayed.  Every output vs its
        reference bitwise on every rank -> max mismatching elements over ranks / calls / reps (0 = race-free in this test)."""
        K, R = a.b2b, a.b2b_reps
        fns, outs = [], []
        allc = list(cases) + [chk]
        M2 = (64 if M != 64 else 128) if a.b2b_mix else 0   # (--b2b-mix: odd calls at another M -- another build, same workspace for tp8v41)
        if M2 and getattr(arms[nm], 'kind', '') in ('v4ft', 'disp') and M2 not in arms[nm].ext:   # (v4ft: per-M exact builds only -> another built M)
            M2 = next((m_ for m_ in sorted(arms[nm].ext) if m_ != M), 0)
        allc2 = []
        if M2:
            c2, k2 = C.tp_cases(sh, M2, dev, a.routing)
            allc2 = list(c2) + [k2]
        for k in range(K):
            mk = M2 if (M2 and k % 2) else M
            _, cx, cids, crw = (allc2 if mk == M2 and M2 else allc)[k % len(allc2 if mk == M2 and M2 else allc)]
            f_, o_ = arms[nm].bind(mk, cx.clone(), cids.clone(), crw.clone())
            fns.append(f_); outs.append(o_)
        refs = []
        for k in range(K):
            torch.cuda.synchronize(); dist.barrier(group=gloo)
            fns[k](); torch.cuda.synchronize()
            refs.append(outs[k].clone())
        torch.cuda.synchronize(); dist.barrier(group=gloo)

        def cmp():
            torch.cuda.synchronize()
            bad = [int((outs[k].view(torch.int16) != refs[k].view(torch.int16)).sum()) for k in range(K)]
            return max(gather_o(max(bad)))

        res = []
        for mode in ('eager', 'drift'):
            worst = 0
            for rep in range(R):
                for o_ in outs:
                    o_.fill_(0)
                torch.cuda.synchronize(); dist.barrier(group=gloo)
                torch.cuda._sleep(int(((rank * 5 + rep * 3) % N) * 30000))   # (rank-skewed start, up to ~(N-1) x 15 us)
                for k in range(K):
                    if mode == 'drift' and rank == rep % N:
                        torch.cuda._sleep(60000)   # (~30 us before each call of one rank)
                    fns[k]()
                worst = max(worst, cmp())
            res.append(f'{mode} {worst}')
        g = torch.cuda.CUDAGraph()
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            fns[0]()
        torch.cuda.current_stream().wait_stream(s_)
        torch.cuda.synchronize(); dist.barrier(group=gloo)
        with torch.cuda.graph(g):
            for k in range(K):
                fns[k]()
        worst = 0
        for rep in range(R):
            for o_ in outs:
                o_.fill_(0)
            torch.cuda.synchronize(); dist.barrier(group=gloo)
            torch.cuda._sleep(int(((rank * 5 + rep * 3) % N) * 30000))
            g.replay()
            worst = max(worst, cmp())
        res.append(f'graph {worst}')
        del g
        torch.cuda.synchronize(); dist.barrier(group=gloo)
        return f' | {nm} b2b(K{K} x{R}{" mixM" + str(M2) if M2 else ""}) mism ' + ' / '.join(res)


    def time_cases(nm, g, row, load):
        """time graph g on every routing case (load(ci) copies case ci into the bound tensors); row[(nm, p)] = per-case values"""
        for ci in range(len(cases)):
            load(ci)
            torch.cuda.synchronize(); dist.barrier(group=gloo)
            for p in a.proto:
                row.setdefault((nm, p), []).append(tm.time(g, p)[0])
                row.setdefault((nm, p + 'm'), []).append(tm.last_mean)
            if CLK:   # (SB_CLK=1: SM clock / temperature / power right after this case's timing, this rank)
                try:
                    row.setdefault((nm, 'clk'), []).append((torch.cuda.clock_rate(dev), torch.cuda.temperature(dev), torch.cuda.power_draw(dev) / 1000.0))
                except Exception as ex:  # noqa: BLE001
                    row.setdefault((nm, 'clk'), []).append((0, 0, 0))

    def load_tp(ci):
        _, cx, cids, crw = cases[ci] if ci >= 0 else chk
        X.copy_(cx); IDS.copy_(cids); RW.copy_(crw)
        if IDSR is not None:
            IDSR[1:].copy_(IDS.unsqueeze(0).expand_as(IDSR[1:]))

    def run_tp(nm, M, row):
        x, ids, rw = X, IDS, RW
        load_tp(-1)
        msg = ''
        if nm in ('ours', 'ours2'):
            fn, out = arms[nm].bind(M, x, ids, rw)
        else:
            moe_out = torch.empty(M, sh.DIM, dtype=torch.bfloat16, device=dev)
            out = torch.empty(M, sh.DIM, dtype=torch.bfloat16, device=dev)
            fn = lambda: trt[1](trt[0](x, ids, rw, moe_out), out)
            import sbtrt
            with sbtrt.autotune_ctx():
                trt[0](x, ids, rw, moe_out)
            torch.cuda.synchronize(); dist.barrier(group=gloo)
        if a.dbg and nm == 'ours' and 'dbg' in arms:   # phase stamps of one cold aligned launch (us from the earliest CTA start of all ranks)
            D = arms['dbg']
            G = D.grid
            if a.trace:
                sys.path.insert(0, str(HERE / 'v77'))
                import xtrace
            dbgt = torch.zeros(xtrace.words(G, 2) if a.trace else G * 32, dtype=torch.int64, device=dev)
            out_d = torch.empty_like(out)
            gd = tm.capture(lambda: D.run_dbg(M, x, ids, rw, out_d, dbgt))   # (graph: no host launch skew between ranks)
            for it in range(3):
                dbgt.zero_()
                if not int(os.environ.get('SB_DBG_NOFLUSH', '0')):
                    tm.flush.zero_()
                dist.barrier(group=gloo); torch.cuda.synchronize()
                tm.align()
                gd.replay()
                torch.cuda.synchronize()
            del gd
            if a.trace:   # (XTrace-style per-warp records of one aligned graph replay: dumped per rank, shown for rank 0)
                tp = Path(os.environ.get('SB_BUILD', '/tmp')) / f'xt_{a.model}_tp{N}_m{M}_r{rank}{"_" + kind if kind != "v77" else ""}{os.environ.get("SB_XT_TAG", "")}.json'
                xtrace.dump(dbgt, G, dict(kernel=kind, m=M, rank=rank, label=a.flags, xt_mask=a.xt_mask), tp, fmt=2)
                dist.barrier(group=gloo)
                if rank == 0:
                    import json as _j
                    xtrace.show(_j.loads(tp.read_text()))
                d = torch.zeros(G, 32, dtype=torch.float64)
            else:
                d = dbgt.view(G, 32).cpu().double()
            t0_ = d[:, 0].min()
            f = lambda k: ((d[:, k][d[:, k] > 0] - t0_) / 1e3)
            keys = (26, 1, 11, 3, 4, 25, 27, 5, 12, 13, 6, 21, 20, 23, 14, 15, 7) if kind != 'v77' else (25, 26, 27, 2, 1, 28, 8, 29, 9, 11, 10, 4, 3, 12, 13, 14, 15, 5, 6, 7)
            row_ = {k: (float(f(k).median()), float(f(k).max())) for k in keys if (d[:, k] > 0).any()}
            sums = {k: float(d[:, k].median() / 1e3) for k in range(16, 24)}
            allr = gather_o((float(t0_), row_, sums))
            if rank == 0:
                if kind == 'v77': log('  (v77 stamps: 25 init 26 pdl 27 gate 2 quant 1 tables 28 tma1 8 x_ready 29 b1 9 mma1 11 1stFC1epi 10 mmaFC1end 4 fc1epiEnd 3 prodDone 12-15 epi 5 roles 6 T1done 7 end)')
                log(f'  DBG M={M} case {chk[0]} med/max us from the rank\'s earliest CTA start (26 pdl 1 tables 11 1stFC1epi 3 fc1pushed 4 lastH '
                    f'25 1stFC2 27 mmaExit 5 roles 12 T1rdy 13 T1item 6 T1done [TAILB: 21 pub 20 1stOwnWait 23 1stOwnPair] 14 ownerRed 15 AGrecv 7 end)')
                for r_, (s0, rw_, sm_) in enumerate(allr):
                    log(f'   r{r_} | ' + ' '.join(f'{k}:{v[0]:.1f}/{v[1]:.1f}' for k, v in rw_.items()) +
                        ' | sums16-23 ' + ' '.join(f'{v:.1f}' for v in sm_.values()))
        if a.check:
            msg += check_tp(nm, fn, out, x, ids, rw, bf16w=(akind(nm, M) == 'v77' or nm == 'trt'))
            if nm in ('ours', 'ours2'):
                ref_out[nm] = out.clone()
                if len(ref_out) == 2:   # (paired builds: bitwise identical outputs on every rank)
                    bad = int((ref_out['ours'].view(torch.int16) != ref_out['ours2'].view(torch.int16)).sum())
                    msg += f' | ours2_vs_ours mism {max(gather_o(bad))}'
        if a.b2b and nm in ('ours', 'ours2'):
            msg += b2b_test(nm, M)
        for _ in range(int(os.environ.get('SB_PRE_EAGER', '0'))):   # (test hook: eager calls before the capture)
            fn()
        torch.cuda.synchronize()
        g = tm.capture(fn)
        if PROBE:
            msg += f' | probe-before {nm}: ' + probe_all()
        HOG = os.environ.get('SB_HOG', '')   # (H1 test) 'ctas,busy' resident spinner on a side stream during this arm's timing
        if HOG:
            import smprobe
            smprobe.hog_start(dev, *[int(v) for v in HOG.split(',')])
        time_cases(nm, g, row, load_tp)
        if HOG:
            smprobe.hog_stop()
        if PROBE:
            msg += f' | probe-after {nm}: ' + probe_all()
        if a.check and nm in ref_out:   # (post-timing: the check case once more through the graph, bitwise vs the pre-timing output)
            load_tp(-1); g.replay(); torch.cuda.synchronize()
            msg += f' | post_mism {max(gather_o(int((out.view(torch.int16) != ref_out[nm].view(torch.int16)).sum())))}'
        del g
        return msg

    def run_ep(nm, M, row):
        T = M // N
        op, wts, buf = ep[nm]
        x_bf = X[rank * T:(rank + 1) * T]

        def load_ep(ci):
            load_tp(ci)
            buf.topk_idx[:T].copy_(IDS[rank * T:(rank + 1) * T].to(torch.int64)); buf.topk_weights[:T].copy_(RW[rank * T:(rank + 1) * T])
        load_ep(-1)
        y = torch.empty(T, sh.DIM, dtype=torch.bfloat16, device=dev)
        xq, xs = buf.x[:T], buf.x_sf[:T]
        msg = ''
        if a.check:
            from deep_gemm.utils import per_token_cast_to_fp8
            r8, rs = per_token_cast_to_fp8(x_bf, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            epq(x_bf, xq, xs); torch.cuda.synchronize()
            bad = int((xq.view(torch.uint8) != r8.view(torch.uint8)).sum()) + int((xs.view(torch.int32) != rs.view(torch.int32)).sum())
            msg = f' | {nm} quant_mism {max(gather_o(bad))}'
        fn = lambda: (epq(x_bf, xq, xs), op(y, *wts, buf, recipe=(1, 1, 32), activation='swiglu', activation_clamp=10.0,
                                              activation_alpha=1.0, fast_math=True))
        torch.cuda.synchronize(); dist.barrier(group=gloo)
        g = tm.capture(fn)
        if PROBE:
            msg += f' | probe-before {nm}: ' + probe_all()
        HOG = os.environ.get('SB_HOG', '')
        if HOG:
            import smprobe
            smprobe.hog_start(dev, *[int(v) for v in HOG.split(',')])
        time_cases(nm, g, row, load_ep)
        if HOG:
            smprobe.hog_stop()
        del g
        return msg

    if 'ours2' in arms and 'ours2' not in a.arms:
        a.arms.append('ours2')
    order = [n for n in ('ours', 'ours2', 'trt', 'gc', 'mega') if n in a.arms]
    if a.swap and 'ours2' in order:
        i, j = order.index('ours'), order.index('ours2'); order[i], order[j] = order[j], order[i]
    if a.order == 'rev':
        order = order[::-1]
    BLOAT, BLOAT_IT = [], [0]
    for M in Ms:
        row = {}
        ref_out = {}
        if a.routing == 'real' and M > C.real_rows(sh.model, dev)[0].shape[0]:
            log(f'SKIP M={M}: real trace has {C.real_rows(sh.model, dev)[0].shape[0]} rows')
            continue
        if os.environ.get('SB_BLOAT'):   # (diagnostic) grow this process' mapped device memory before each M: 'KB_per_alloc,count' raw cudaMallocs
            kb_, cnt_ = (int(v) for v in os.environ['SB_BLOAT'].split(','))
            BLOAT_IT[0] += 1
            if BLOAT_IT[0] > 1 or int(os.environ.get('SB_BLOAT_FIRST', '0')):
                for _ in range(cnt_):
                    BLOAT.append(rt.malloc(kb_ * 1024))
            torch.cuda.synchronize()
            log(f'  BLOAT: {len(BLOAT)} extra cudaMallocs of {kb_} KB = {len(BLOAT) * kb_ / 1048576:.1f} GB; mem_get_info free {torch.cuda.mem_get_info(dev)[0] / 2**30:.1f} GB')
        cases, chk = C.tp_cases(sh, M, dev, a.routing)
        X, IDS, RW = (chk[i].clone() for i in (1, 2, 3))
        IDSR = None
        if int(os.environ.get('SB_IDSREP', '0')) > 1:   # (test: TABDBG >= 2 kernels read replica b % R of the ids, laid out after IDS)
            R_ = int(os.environ['SB_IDSREP']); IDSR = torch.empty((R_,) + tuple(IDS.shape), dtype=IDS.dtype, device=IDS.device)
            IDSR[0].copy_(IDS); IDS = IDSR[0]
        st_ = [C.ep_stats(sh, c[2]) for c in cases]
        rr, emax, U = (sum(v[i] for v in st_) / len(st_) for i in range(3))
        msg = f' | {a.routing} EPrecv max/mean {rr:.2f} U {U:.0f} maxload {emax:.0f}' + (f' ({len(cases)} cases, check on {chk[0]})' if len(cases) > 1 else '')
        for nm in order:
            if nm in ('gc', 'mega'):
                if M % N == 0:
                    msg += run_ep(nm, M, row)
            else:
                msg += run_tp(nm, M, row)
        line = f'RES M={M:4d} T={M // N:3d}'
        rng = ''
        clk = {nm: v for (nm, p), v in row.items() if p == 'clk'}
        for (nm, p), v in row.items():
            if p == 'clk':
                continue
            line += f' | {nm}.{p} {sum(v) / len(v):7.1f}'
            if len(v) > 1 and not p.endswith('m'):
                rng += f' {nm}.{p}[{min(v):.1f}..{max(v):.1f}]'
        if clk:   # (SB_CLK) min SM MHz / max temp C / max W over ranks and cases, per arm
            allc = gather_o(clk)
            for nm in clk:
                vv = [x for r_ in allc for x in r_[nm]]
                msg += f' | {nm} clk {min(x[0] for x in vv)}-{max(x[0] for x in vv)} MHz T {max(x[1] for x in vv)} C P {max(x[2] for x in vv):.0f} W'
        log(line + msg + (' | range' + rng if rng else ''))
    torch.cuda.synchronize(); dist.barrier(group=gloo)
    log(f'done {time.time() - t0:.0f} s')
    dist.destroy_process_group()
    os._exit(0)   # (gc-moe processes can hang at exit with live symmetric memory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['dsv4f', 'dsv41'], required=True)
    ap.add_argument('--tp', type=int, choices=[4, 8], required=True)
    ap.add_argument('--ms', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256, 384, 512])
    ap.add_argument('--arms', nargs='+', default=['ours'])
    ap.add_argument('--ours', default='')
    ap.add_argument('--proto', nargs='+', default=['b'])
    ap.add_argument('--flags', default='')
    ap.add_argument('--oneshot-maxm', type=int, default=32)
    ap.add_argument('--check', type=int, default=1)
    ap.add_argument('--windows', type=int, default=30); ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--cold-iters', type=int, default=50)
    ap.add_argument('--sleep-cycles', type=int, default=200000)
    ap.add_argument('--order', default='fwd')
    ap.add_argument('--routing', default='real', choices=['uniform', 'real', 'trace'])
    ap.add_argument('--trt-ipad', type=int, default=0)
    ap.add_argument('--trtgen-cmp', type=int, default=0)   # (v77: count mismatches vs this flashinfer's trtllm-gen; 0.7.3 quantizes h differently)
    ap.add_argument('--flags2', default=None)
    ap.add_argument('--src2', default=None)   # (tp8v41 lineage) ours2 from another source file in the same host dir (same-run source A/B)
    ap.add_argument('--oneshot-maxm2', type=int, default=0)
    ap.add_argument('--ours2-kind', default='')   # (ours2 from another kernel family, e.g. v4ft for dsv4f: cross-kind same-run A/B)
    ap.add_argument('--dispatch', default='')     # (v4ft ours) per-M extra flags 'M:-DA=1,-DB=2 M2:..'
    ap.add_argument('--dispatch2', default='')    # (v4ft ours2) per-M extra flags
    ap.add_argument('--swap', type=int, default=0)
    ap.add_argument('--dbg', type=int, default=0)
    ap.add_argument('--b2b', type=int, default=0)   # back-to-back correctness test: K calls per burst
    ap.add_argument('--b2b-reps', type=int, default=4)
    ap.add_argument('--b2b-mix', type=int, default=0)
    ap.add_argument('--trace', type=int, default=0)   # (with --dbg: v77 XTrace records instead of stamps)
    ap.add_argument('--xt-mask', type=int, default=1)
    a = ap.parse_args()
    a.flags = a.flags.replace(',', ' ')
    a.flags2 = a.flags2.replace(',', ' ') if a.flags2 is not None else None
    if a.src2 is not None and a.flags2 is None:
        a.flags2 = ''
    a.rdv = tempfile.mktemp(prefix='sb_rdv_', dir=os.environ.get('SB_BUILD', '/tmp'))
    torch.multiprocessing.spawn(worker, args=(a,), nprocs=a.tp, join=True)


if __name__ == '__main__':
    main()
