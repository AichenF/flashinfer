"""HBM roofline of one DSV4-Flash MoE layer at TP4 per M from the captured routing: unique routed experts (production router + gate),
weight bytes one rank must stream (routed MXFP4 w13 / w2 slices + UE8M0 scales, the FP8 shared expert slice), and the bound
time at a given HBM rate.  usage: python3 b2_roofline.py --model x --cap x --out x"""
import argparse
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--cap', required=True); ap.add_argument('--out')
    ap.add_argument('--layer', type=int, default=20)
    args = ap.parse_args()
    import b2_common as C
    dev = torch.device('cuda', 0)
    torch.cuda.set_device(dev)
    wm = C.weight_map(args.model)
    router, bias = C.load_router(args.model, wm, args.layer, dev)
    ms = [1, 2, 4, 6, 8, 16, 20, 24, 32, 48, 64, 96, 128]
    xs = C.capture_pool(args.cap, ms, dev)
    P = C.Prod(None, None, router, bias, None, None)
    MiB = 1 << 20
    per_exp = (1024 * 2048 + 1024 * 128) + (4096 * 256 + 4096 * 16)   # w13 + sf, w2 + sf (bytes per rank)
    shared = 1024 * 4096 + 4096 * 512 + (8 * 32 + 32 * 4) * 4         # FP8 gate_up + down slices (+ per-128x128 scales)
    print('M  uniqE  bytes/rank(MiB)  t@6.9TB/s(us)  t@6.0TB/s(us)', flush=True)
    for m in ms:
        lg = P.logits(xs[m])
        w, i = P.topk(lg)
        u = int(torch.unique(i.flatten()).numel())
        byt = u * per_exp + shared
        print(f'{m:4d} {u:5d} {byt / MiB:9.1f} {byt / 6.9e6:9.1f} {byt / 6.0e6:9.1f}', flush=True)
    print('ROOFLINE_DONE')


if __name__ == '__main__':
    main()
