"""stress_v77.py: high-count back-to-back correctness stress of our kernels on N B200 GPUs: v77 (DSV4-Flash, --model dsv4f) and the
tp8v41 lineage (DSV4.1-Flash, --model dsv41 --mode routed; --flags e.g. -DTPB2_F2H=1).

  --mode prod    the PRODUCTION builds: pristine tpmoe_b2.cu (tpmoe_b200/src @502e1d2 = v77/tpmoe_b2_prod.cu), tpb2_dispatch.V77 flags
                 (TPB2_SPEC=1): fused gate from router logits (or the in-kernel router where production uses it, M 5..64 with a probed
                 split), fused shared expert (SharedWeights), 1.5 x routed + shared, in-kernel LL all-reduce.  TP4 (the package's NDEV).
  --mode routed  the suite_b200 routed-only builds (arm_v77: SPEC=2 + IDV + LLS + XRLX), given top-k ids / weights.

Per M: --cases distinct inputs; reference = one sync- and barrier-separated call per case.  Then --rounds rounds of --calls back-to-back
eager calls (cases alternating, own output per call, rank-skewed starts; every 8th round one rank sleeps before each of its calls) and
one CUDA graph of --calls calls replayed --rounds times.  Every output vs its reference, bitwise, on every rank: mismatching calls /
rows / elements (max over ranks).  usage: torchrun-free (spawns): python stress_v77.py --mode prod --tp 4 [--ms ..] [--rounds 1500]"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'v77'))
import sbcommon as C   # noqa: E402


def worker(rank, a):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    N = a.tp
    dist.init_process_group('nccl', init_method='file://' + a.rdv, world_size=N, rank=rank, device_id=dev)
    gloo = dist.new_group(backend='gloo')
    log = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)
    sh = C.Shape(a.model, N)
    rt = C.CudaRT(); rt.ck(rt.lib.cudaSetDevice(rank))
    ipc = lambda n: C.ipc_table(rt, n, rank, N, gloo)
    w1, w2 = C.synth_slices(sh, rank, dev)
    raw = C.quantize_slices(sh, w1, w2)
    del w1, w2
    t0 = time.time()
    if a.model == 'dsv41':   # (DSV4.1-Flash: the tp8v41-lineage arm, routed; --flags adds build flags, e.g. -DTPB2_F2H=1)
        import arm_tp8v41
        assert a.mode == 'routed', 'DSV4.1: routed builds only'
        Ms = a.ms or [64, 128, 256, 384, 512]
        arm = arm_tp8v41.Arm(sh, rank, dev, gloo, ipc, raw, Ms, extra=tuple(a.flags.split()))
        fl = [x for x in arm.fl[Ms[-1]] if any(k in x for k in ('F2H', 'ACTX', 'NMAX', 'LLS', 'GATHER'))]
    else:
        import sbtrt
        import arm_v77
        import tpb2
        import tpb2_dispatch as D
        trtW = sbtrt.prep_weights(sh, raw)
    if a.model == 'dsv41':
        pass
    elif a.mode == 'prod':
        assert N == 4, 'the production package is TP4'
        Ms = a.ms or sorted({hi for lo, hi, _ in D.V77} | {lo for lo, hi, _ in D.V77} | {(lo + hi) // 2 for lo, hi, _ in D.V77})
        flags = {M: D.build_flags(M) for M in Ms}
        uniq = sorted(set(flags.values()))
        for i, f in enumerate(uniq):
            if i % N == rank:
                arm_v77.load_ext(f, 'tpmoe_b2_prod.cu')
        dist.barrier(group=gloo)
        ext = {f: arm_v77.load_ext(f, 'tpmoe_b2_prod.cu') for f in uniq}
        state, W, SW = {}, {}, {}
        g = torch.Generator(device=dev).manual_seed(4242 + rank)
        f8 = lambda *s: (torch.randn(*s, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
        p2 = lambda *s: torch.exp2(torch.randint(-9, -5, s, generator=g, device=dev).float())
        gu_w, gs, dn_w, ds = f8(1024, 4096), p2(8, 32), f8(4096, 512), p2(32, 4)
        rwt = (torch.randn(256, 4096, generator=g, device=dev) * 0.02).to(torch.bfloat16)
        torch.manual_seed(1)   # (same router weights on every rank: production's replicated gate)
        rwt.copy_((torch.randn(256, 4096, device=dev) * 0.02).to(torch.bfloat16))
        for f, E in ext.items():
            st = tpb2.RankState(E, dev, N, rank)
            st.link_ptrs(ipc(tpb2.RankState.RS_BYTES(E)), ipc(tpb2.RankState.AG_BYTES(E)))
            state[f] = st
            W[f] = tpb2.Weights(E, trtW['w13'], trtW['s13'], trtW['w2'], trtW['s2'])
            SW[f] = tpb2.SharedWeights(E, gu_w, gs, dn_w, ds)
        gbias = (torch.randn(256, device=dev, generator=torch.Generator(device=dev).manual_seed(7)) * 0.05).float()
    elif a.kern == 'disp':   # (DSV4-Flash dispatch arm: v77 below SB_V4FT_MINM (384), v4ft from it; --mix2 M2 alternates families)
        import arm_disp
        Ms = a.ms or [384, 512]
        arm = arm_disp.Arm(sh, rank, dev, gloo, ipc, raw, trtW, sorted(set(Ms) | ({a.mix2} if a.mix2 else set())), extra=tuple(a.flags.split()))
        fl = [arm.name] + [x for x in (arm.v4ft.fl[Ms[-1]] if arm.v4ft and Ms[-1] in arm.v4ft.fl else ()) if any(k in x for k in ('ACTX', 'T1PW', 'NMAX', 'LLS', 'F2H'))]
    elif a.kern == 'v4ft':
        import arm_disp
        import arm_v4ft
        Ms = a.ms or [384, 512]
        arm = arm_v4ft.Arm(sh, rank, dev, gloo, ipc, raw, sorted(set(Ms) | ({a.mix2} if a.mix2 else set())), extra=tuple(a.flags.split()), per_m=arm_disp.V4FT_PER_M)
        fl = [x for x in arm.fl[Ms[-1]] if any(k in x for k in ('ACTX', 'T1PW', 'NMAX', 'LLS', 'F2H'))]
    else:
        Ms = a.ms or [256, 384, 512]
        arm = arm_v77.Arm(sh, rank, dev, gloo, ipc, trtW, Ms, extra=tuple(a.flags.split()))
        fl = []
    log(f'[{time.strftime("%T")}] {a.model} flags of the last build (selected): {fl if a.mode == "routed" else "production table"}')
    log(f'[{time.strftime("%T")}] {a.mode} TP{N} builds ready {time.time() - t0:.0f} s; M {Ms}; rounds {a.rounds} x {a.calls} calls + graph, {a.cases} cases')

    tot = dict(calls=0, bad_calls=0, bad_el=0)
    def mk_cases(M):
        cases = []
        for c in range(a.cases):
            gg = torch.Generator(device=dev).manual_seed(1000 * M + c)   # (same inputs on every rank: the TP layer input)
            x = (torch.randn(M, sh.DIM, generator=gg, device=dev) * 0.5).to(torch.bfloat16)
            if a.mode == 'prod':
                glog = torch.randn(M, 256, generator=gg, device=dev).float()
                cases.append((x, glog))
            else:
                sc = torch.rand(M, sh.NEXP, generator=gg, device=dev)
                ids = sc.topk(6, dim=1).indices.to(torch.int32).contiguous()
                rw = torch.softmax(torch.randn(M, 6, generator=gg, device=dev), dim=1).float().contiguous()
                cases.append((x, ids, rw))
        return cases

    for M in Ms:
        cases = mk_cases(M)
        M2 = a.mix2 if (a.mix2 and a.mode == 'routed' and a.mix2 != M) else 0
        cases2 = mk_cases(M2) if M2 else None
        if a.mode == 'prod':
            f = flags[M]
            st, Wm, SWm = state[f], W[f], SW[f]
            parts = D.in_kernel_router(M)
            router = (rwt, parts) if (parts and not a.no_router) else None

            def mk(c, out):
                x, glog = cases[c]
                return lambda: tpb2.run(st, Wm, x, out, glog=glog, gbias=gbias, SW=SWm, factor=1.5, router=router)
        else:
            def mk(c, out, Mx=None):
                x, ids, rw = (cases2 if Mx else cases)[c]
                return arm.bind(Mx or M, x, ids, rw, out=out)[0]
        K = a.calls
        mk_ = [M2 if (M2 and k % 2) else 0 for k in range(K)]   # (--mix2: odd calls at M2)
        outs = [torch.empty(mk_[k] or M, sh.DIM, dtype=torch.bfloat16, device=dev) for k in range(K)]
        fns = [(mk(k % a.cases, outs[k], mk_[k]) if mk_[k] else mk(k % a.cases, outs[k])) for k in range(K)]
        refs, refs2 = [], []
        for c in range(a.cases * (2 if M2 else 1)):
            Mx = M2 if c >= a.cases else 0
            o = torch.empty(Mx or M, sh.DIM, dtype=torch.bfloat16, device=dev)
            fn = mk(c % a.cases, o, Mx) if Mx else mk(c, o)
            for _ in range(2):
                torch.cuda.synchronize(); dist.barrier(group=gloo)
                fn()
            torch.cuda.synchronize()
            (refs2 if Mx else refs).append(o.clone())
        # reference sanity: every rank holds the same all-reduced output; deterministic over two separated calls
        same = []
        for c in range(a.cases):
            allr = [torch.empty_like(refs[c]) for _ in range(N)]
            dist.all_gather(allr, refs[c])
            same.append(all(torch.equal(allr[0], v) for v in allr))
        refk = [(refs2 if mk_[k] else refs)[k % a.cases] for k in range(K)]

        def check(st_):
            torch.cuda.synchronize()
            bad_rows = [int((outs[k].view(torch.int16) != refk[k].view(torch.int16)).any(dim=1).sum()) for k in range(K)]
            bad_el = [int((outs[k].view(torch.int16) != refk[k].view(torch.int16)).sum()) for k in range(K)]
            st_['calls'] += K
            st_['bad_calls'] += sum(1 for v in bad_el if v)
            st_['bad_rows'] += sum(bad_rows)
            st_['bad_el'] += sum(bad_el)
            if any(bad_el) and len(st_['ex']) < 3:
                k = next(i for i, v in enumerate(bad_el) if v)
                rows = (outs[k].view(torch.int16) != refk[k].view(torch.int16)).any(dim=1).nonzero().flatten().tolist()
                st_['ex'].append((k, rows[:8]))

        res = {}
        for mode in ('eager', 'graph'):
            st_ = dict(calls=0, bad_calls=0, bad_rows=0, bad_el=0, ex=[])
            g = None
            if mode == 'graph':
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
                torch.cuda.synchronize(); dist.barrier(group=gloo)
            t1 = time.time()
            for r in range(a.rounds):
                if r % 50 == 0:
                    dist.barrier(group=gloo)
                for o in outs:
                    o.fill_(0)
                torch.cuda._sleep(int(((rank * 7 + r * 3) % (2 * N)) * 4000))   # (rank-skewed start, ~0..15 us)
                if mode == 'graph':
                    g.replay()
                else:
                    for k in range(K):
                        if r % 8 == 7 and rank == (r // 8) % N:
                            torch.cuda._sleep(20000)   # (one rank drifts: ~10 us before each of its calls)
                        fns[k]()
                check(st_)
            dt = time.time() - t1
            del g
            torch.cuda.synchronize()
            alls = [None] * N
            dist.all_gather_object(alls, st_, group=gloo)
            res[mode] = dict(calls=st_['calls'], bad_calls=max(v['bad_calls'] for v in alls), bad_rows=max(v['bad_rows'] for v in alls),
                             bad_el=max(v['bad_el'] for v in alls), ex=[(r_, v['ex']) for r_, v in enumerate(alls) if v['ex']][:2], s=dt)
            tot['calls'] += st_['calls']
            tot['bad_calls'] += res[mode]['bad_calls']
            tot['bad_el'] += res[mode]['bad_el']
        msg = ' | '.join(f"{m_} {v['calls']} calls ({v['s']:.0f} s): bad calls {v['bad_calls']} rows {v['bad_rows']} el {v['bad_el']}" +
                         (f" ex {v['ex']}" if v['ex'] else '') for m_, v in res.items())
        log(f'[{time.strftime("%T")}] M={M:4d}' + (f' mix M{M2}' if M2 else '') + ' ' + ('router ' if a.mode == 'prod' and router else '') +
            f'ref ranks_same {all(same)} | ' + msg)
    log(f'TOTAL calls per rank {tot["calls"]}, bad calls (max over ranks, summed) {tot["bad_calls"]}, bad elements {tot["bad_el"]}')
    torch.cuda.synchronize(); dist.barrier(group=gloo)
    dist.destroy_process_group()
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['prod', 'routed'], required=True)
    ap.add_argument('--tp', type=int, default=4)
    ap.add_argument('--ms', type=int, nargs='*', default=None)
    ap.add_argument('--rounds', type=int, default=1500)
    ap.add_argument('--calls', type=int, default=8)
    ap.add_argument('--cases', type=int, default=2)
    ap.add_argument('--no-router', type=int, default=0)
    ap.add_argument('--model', default='dsv4f', choices=['dsv4f', 'dsv41'])
    ap.add_argument('--flags', default='')
    ap.add_argument('--kern', default='v77', choices=['v77', 'v4ft', 'disp'])   # (dsv4f --mode routed: kernel family / dispatch arm)
    ap.add_argument('--mix2', type=int, default=0)   # (routed: odd calls at this M -- another build / kernel family, same stream)
    a = ap.parse_args()
    a.flags = a.flags.replace(',', ' ')
    a.rdv = tempfile.mktemp(prefix='stress_rdv_', dir=os.environ.get('SB_BUILD', '/tmp'))
    mp.spawn(worker, args=(a,), nprocs=a.tp, join=True)


if __name__ == '__main__':
    main()
