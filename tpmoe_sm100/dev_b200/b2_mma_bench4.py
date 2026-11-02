"""tcgen05 block-scaled MMA rate with the kernel's clean issue shape (b2_mma_bench.cu mma_bench4): ns per MMA by MMAs per round (commit),
commit-wait depth (0 = never wait), N; 1-CTA M128 vs CTA pair M256.  usage: python3 b2_mma_bench4.py --model x --cap x --out x"""
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
    for pair in (0, 1):
        for n in (8, 16, 32) if not pair else (16, 32):
            for per in (8, 16, 32):
                for depth in (0, 1, 2, 4):
                    cyc = X.run4(pair, per, n, rounds, depth, sms - sms % 2).double()
                    torch.cuda.synchronize()
                    cyc = cyc[cyc > 0]
                    ns = float(cyc.median()) / clk / (rounds * per)
                    print(f'{"pair M256" if pair else "cta1 M128"} N{n:3d} per{per:3d} depth{depth} : {ns:6.1f} ns/MMA  {ns * per:7.1f} ns/round', flush=True)
    print('MMA_BENCH4_DONE')


if __name__ == '__main__':
    main()
