"""Weight-stream ceiling by row shape: w13-like [rows][2048 B] vs w2-like [rows][256 B] (b2_tma_bench.cu, 16U4).  usage: ... --model x --cap x --out x"""
import argparse, os
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.parse_args()
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
(SRV / 'build_tma_bench').mkdir(exist_ok=True)
X = load(name='b2_tma_bench', sources=[str(SRV / 'b2_tma_bench.cu')], build_directory=str(SRV / 'build_tma_bench'),
         extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'], extra_ldflags=['-lcuda'], verbose=False)
sms = torch.cuda.get_device_properties(0).multi_processor_count
w = torch.randint(0, 255, (1 << 30,), dtype=torch.uint8, device='cuda')
for rowb in (2048, 256):
    for kc in (1, 2, 4):
        if kc * 64 > rowb: continue
        for ns in (2, 3, 4, 6):
            if ns * kc * 16384 + 2048 > 232448: continue
            print(f'rowb {rowb:5d} KC{kc} NS{ns} : {X.run(w, 0, kc, ns, sms, 1, 2, rowb):7.0f} GB/s', flush=True)
print('TMA2_DONE')
