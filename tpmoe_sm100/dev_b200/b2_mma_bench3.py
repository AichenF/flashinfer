"""Which per-round step costs ~250 ns in the tcgen05 MMA issue loop (b2_mma_bench.cu mma_bench3)?  ns per MMA by MMAs per round (per),
commit period (rounds per commit), and mode bits (1 no fence, 2 one elect, 4 no mbarrier waits); 1-CTA (M128) and CTA pair (M256).
usage: python3 b2_mma_bench3.py --model x --cap x --out x"""
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
    clk = 1.965
    rounds = 2048

    def one(pair, n, per, cper, mode):
        cyc = X.run3(pair, n, per, rounds, cper, mode, sms - sms % 2).double()
        torch.cuda.synchronize()
        cyc = cyc[cyc > 0]
        ns = float(cyc.median()) / clk / (rounds * per)
        print(f'{"pair M256" if pair else "cta1 M128"} N{n:3d} per{per:3d} cper{cper:2d} mode{mode} : {ns:6.1f} ns/MMA  {ns * per:7.1f} ns/round', flush=True)
    for pair in (0, 1):
        n = 32
        for per in (4, 8, 16, 32, 64):
            one(pair, n, per, 1, 0)
        for mode in (1, 2, 3, 4, 7):
            one(pair, n, 8, 1, mode)
        for cper in (2, 4, 8):
            one(pair, n, 8, cper, 0)
            one(pair, n, 8, cper, 7)
        for per in (4, 16, 64):
            one(pair, n, per, 1, 7)
    print('MMA_BENCH3_DONE')


if __name__ == '__main__':
    main()
