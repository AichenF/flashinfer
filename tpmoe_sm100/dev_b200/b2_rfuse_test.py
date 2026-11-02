"""b2_rfuse.cu: our router GEMM (tcgen05 kind::f16 K-part chains, M = 64 swapped roles) vs production torch.mm(x, w.t(), out_dtype=fp32)
(cuBLAS split-K), bitwise, per M (router_splits table); + the time per call vs cuBLAS.  usage: python3 b2_rfuse_test.py --model M --cap C --out O"""
import argparse
import os
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402
from router_splits import router_parts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out')
    ap.add_argument('--ms', type=int, nargs='+', default=[5, 6, 8, 12, 16, 20, 24, 32, 48, 64])
    args = ap.parse_args()
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    (SRV / 'build_rfuse').mkdir(exist_ok=True)
    X = load(name='b2_rfuse', sources=[str(SRV / 'b2_rfuse.cu')], build_directory=str(SRV / 'build_rfuse'),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'],
             extra_ldflags=['-lcuda'], verbose=False)
    dev = torch.device('cuda', 0)
    wm = C.weight_map(args.model)
    router, bias = C.load_router(args.model, wm, 20, dev)
    router = router.contiguous()
    xs = C.capture_pool(args.cap, args.ms, dev)
    for m in args.ms:
        x = xs[m].contiguous()
        sp = router_parts(m)
        bounds, k = [], 0
        for kn in sp:
            bounds += [k, kn]; k += kn
        S = len(sp)
        part = torch.zeros(S, m, 256, dtype=torch.float32, device=dev)
        dbg = torch.zeros(128, 16, dtype=torch.float32, device=dev)
        X.run(x, router, part, bounds, dbg, 0)
        torch.cuda.synchronize()
        lg = part[0].clone()
        for p in range(1, S):
            lg = lg + part[p]
        ref = torch.mm(x, router.t(), out_dtype=torch.float32)
        ne = int((lg != ref).sum())
        msg = f'M {m:3d} parts {sp}: == cuBLAS {ne == 0} mismatches {ne}/{lg.numel()} max|d| {float((lg - ref).abs().max()):.3e}'
        if ne:   # layout probe: which TMEM lane / column holds token t, expert j (block 0 = experts 0..15, part 0)
            p0 = torch.mm(x[:, bounds[0]:bounds[0] + bounds[1]], router[:16, bounds[0]:bounds[0] + bounds[1]].t(), out_dtype=torch.float32)
            hits = []
            for t in range(min(m, 4)):
                for j in (0, 1):
                    loc = (dbg == p0[t, j]).nonzero().tolist()
                    hits.append(f't{t}e{j}@{loc[:2]}')
            msg += '  probe ' + ' '.join(hits)
        # time per call: ours (no PDL) vs cuBLAS torch.mm, 200 calls each
        for _ in range(5):
            X.run(x, router, part, bounds, None, 0); torch.mm(x, router.t(), out_dtype=torch.float32)
        torch.cuda.synchronize()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        for _ in range(200):
            X.run(x, router, part, bounds, None, 0)
        e1.record()
        for _ in range(200):
            torch.mm(x, router.t(), out_dtype=torch.float32)
        e2.record()
        torch.cuda.synchronize()
        msg += f'  us/call ours {e0.elapsed_time(e1) * 5:.2f} cuBLAS {e1.elapsed_time(e2) * 5:.2f}'
        print(msg, flush=True)
    print('RFUSE_DONE')


if __name__ == '__main__':
    main()
