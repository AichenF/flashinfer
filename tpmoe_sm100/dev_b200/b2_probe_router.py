"""Which kernels does the production router GEMM run on B200, per M?  M <= 4: dsv3_router_gemm (JIT); else torch.mm(x, w.t(),
out_dtype=float32) (cuBLAS).  Prints the kernel names (split-K? reduction kernel?) and checks whether a plain fp32 K-sequential
reference or an fp64 reference matches cuBLAS bit for bit.  usage: python3 b2_probe_router.py --model MODEL --cap CAP --out OUT"""
import argparse
import sys
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--cap', required=True); ap.add_argument('--out')
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 2, 4, 5, 6, 8, 16, 20, 24, 32, 48, 64, 96, 128])
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    wm = C.weight_map(args.model)
    router, bias = C.load_router(args.model, wm, 20, dev)
    xs = C.capture_pool(args.cap, args.ms, dev)
    P = C.Prod(None, dict(inter=512), router, bias, None, None)
    from torch.profiler import profile, ProfilerActivity
    for m in args.ms:
        x = xs[m]
        for _ in range(3):
            P.logits(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            lg = P.logits(x)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type.name == 'CUDA']
        # bitwise checks against simple references
        r64 = (x.double() @ router.double().t()).float()
        print(f'M {m:4d}: kernels {names}  max|lg - fp64 ref| {float((lg - r64).abs().max()):.3e}', flush=True)
    print('PROBE_DONE')


if __name__ == '__main__':
    main()
