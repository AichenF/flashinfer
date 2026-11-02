"""tcgen05 block-scaled MMA rate on B200 (b2_mma_bench.cu): ns per MMA (M128 x N x K32) by N, MMAs per commit, SF copies, commit
pipelining depth.  usage: python3 b2_mma_bench.py --model x --cap x --out x"""
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
    (SRV / 'build_mma_bench').mkdir(exist_ok=True)
    X = load(name='b2_mma_bench', sources=[str(SRV / 'b2_mma_bench.cu')], build_directory=str(SRV / 'build_mma_bench'),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'], verbose=False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    clk = 1.965   # GHz (the run's SM clock)
    rounds = 2000
    print('atype N per cp depth grid ilv : ns/MMA (median CTA)', flush=True)
    def one(atype, n, per, cp, depth, grid, ilv):
        cyc = X.run(n, per, rounds, cp, depth, atype, grid, ilv).double()
        torch.cuda.synchronize()
        ns = float(cyc.median()) / clk / (rounds * per)
        print(f'{"e2m1" if atype == 5 else "e4m3"} N{n:3d} per{per:3d} cp{cp} depth{depth} grid{grid:4d} ilv{ilv} : {ns:6.1f} ns/MMA', flush=True)
    # the key question first: is a chained accumulator the limit (latency) or the issue rate?
    for n in (8, 16, 32):
        for ilv in (1, 2, 4):
            for depth in (1, 4):
                one(5, n, 16, 1, depth, sms, ilv)
    for atype in (5, 0):
        for n in (8, 16, 32, 64):
            for per in (8, 16):
                for cp in (0, 1):
                    for depth in (1, 4):
                        for grid in (1, sms):
                            one(atype, n, per, cp, depth, grid, 1)
    print('cta_group::2 (pair, leader issues M256):', flush=True)
    for n in (16, 32, 64):
        for per in (8, 16):
            for cp in (0, 1):
                for depth in (1, 4):
                    for grid in (2, sms - sms % 2):
                        cyc = X.run2(n, per, rounds, cp, depth, grid).double()
                        torch.cuda.synchronize()
                        cyc = cyc[cyc > 0]
                        ns = float(cyc.median()) / clk / (rounds * per)
                        print(f'2cta e2m1 N{n:3d} per{per:3d} cp{cp} depth{depth} grid{grid:4d} : {ns:6.1f} ns/MMA (M256)', flush=True)
    print('MMA_BENCH_DONE')


if __name__ == '__main__':
    main()
