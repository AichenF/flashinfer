"""Reverse-engineer the production router GEMM (cuBLAS nvjet split-K + splitKreduce for M >= 5) bitwise: for each M, find a split S of K
(equal parts) whose per-part fp32 GEMMs, summed in order, reproduce torch's result exactly.  usage: python3 b2_router_re.py --model M --cap C --out O"""
import argparse
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402


def lin(x, w):   # the production router GEMM for M > 4 (sglang dsv4 linear_bf16_fp32 default path)
    return torch.mm(x, w.t(), out_dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--cap', required=True); ap.add_argument('--out')
    ap.add_argument('--ms', type=int, nargs='+', default=list(range(5, 65)) + [72, 80, 96, 112, 128])
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    wm = C.weight_map(args.model)
    router, bias = C.load_router(args.model, wm, 20, dev)
    xs = C.capture_pool(args.cap, args.ms, dev)
    K = router.shape[1]
    for m in args.ms:
        x = xs[m]
        full = lin(x, router)
        found = []
        tried = set()
        for S in (4, 3, 2, 5, 6, 8):
            for unit in (64, 128):
                for mode in ('ceil', 'floor'):
                    if mode == 'ceil':
                        kp = -(-K // S); kp = -(-kp // unit) * unit
                        bounds = [(i * kp, min(K, (i + 1) * kp)) for i in range(S) if i * kp < K]
                    else:   # equal parts rounded down to the unit, the remainder in the last part
                        kp = (K // S) // unit * unit
                        if kp == 0: continue
                        bounds = [(i * kp, (i + 1) * kp if i < S - 1 else K) for i in range(S)]
                    key = tuple(bounds)
                    if key in tried or len(bounds) < 2: continue
                    tried.add(key)
                    parts = [lin(x[:, a:b].contiguous(), router[:, a:b].contiguous()) for a, b in bounds]
                    seq = parts[0].clone()
                    for p_ in parts[1:]:
                        seq = seq + p_
                    if torch.equal(seq, full):
                        found.append(f'{[b - a for a, b in bounds]}')
        print(f'M {m:4d}: exact candidates: {found if found else "none"}', flush=True)
    print('RE_DONE')


if __name__ == '__main__':
    main()
