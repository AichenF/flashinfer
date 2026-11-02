"""tcgen05 block-scaled MMA (kind::mxf8f6f4, e2m1 A x e4m3 B, K32): M64 vs M128 ns per MMA (does a 64-row FC1 item stream twice as fast?)
usage: python3 b2_mma_bench6.py --model x --cap x --out x"""
import argparse
import os
from pathlib import Path

import torch

SRV = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out')
    ap.parse_args()
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    (SRV / 'build_mma_bench6').mkdir(exist_ok=True)
    X = load(name='b2_mma_bench6', sources=[str(SRV / 'b2_mma_bench.cu')], build_directory=str(SRV / 'build_mma_bench6'),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'], verbose=False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    clk = 1.965
    rounds = 2000
    def one(tag, n, per, at, bt, depth, grid, rnd):
        a = at | (bt << 8)
        if rnd: a = a - (1 << 32) + (1 << 31) + ((1 << 30) if (at == 5 and bt == 5) else 0)   # (int64 -> int: bit 31 set)
        cyc = X.run(n, per, rounds, 1, depth, a, grid, 1).double()
        torch.cuda.synchronize()
        ns = float(cyc.median()) / clk / (rounds * per)
        rows = n if n > 64 else 128
        print(f'{tag} {"rand" if rnd else "0x3C"} N{n:3d} per{per} depth{depth} grid{grid:4d} : {ns:6.1f} ns/MMA ({ns * 1e3 / rows:7.1f} ps per weight row x K32)', flush=True)
    for rep in range(2):
        for rnd in (1, 0):
            for grid in (1, sms):
                for n in (8, 32):
                    one('cur   A=w e2m1 B=x e4m3', n, 8, 5, 0, 2, grid, rnd)
                    one('fp8w  A=w e4m3 B=x e4m3', n, 8, 0, 0, 2, grid, rnd)
                for n in (128, 256):
                    one('swap  A=x e4m3 B=w e2m1', n, 4, 0, 5, 2, grid, rnd)
                    one('swapf A=x e4m3 B=w e4m3', n, 4, 0, 0, 2, grid, rnd)
    print('MMA_BENCH_DONE')


if __name__ == '__main__':
    main()
