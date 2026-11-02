"""Router M without an equal-split match: is the production result one full-K tcgen05 K16 chain (no split)?  Or a split summed in another
order (reverse / pairwise)?  usage: python3 b2_router_re2.py --model M --cap C --out O"""
import argparse, os, sys
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); args = ap.parse_args()
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
X = load(name='b2_bf16mma_test', sources=[str(SRV / 'b2_bf16mma_test.cu')], build_directory=str(SRV / 'build_bmt'),
         extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a'], verbose=False)
dev = torch.device('cuda', 0)
wm = C.weight_map(args.model)
router, bias = C.load_router(args.model, wm, 20, dev)
ms = [25, 28, 31, 49, 56, 63, 72, 96, 128]
xs = C.capture_pool(args.cap, ms, dev)
for m in ms:
    x = xs[m].contiguous()
    full = torch.mm(x, router.t(), out_dtype=torch.float32)
    chain = X.run_swap(x, router.contiguous(), 0, 4096).t()
    res = [f'full chain {torch.equal(chain, full)}']
    for S in (2, 3, 4, 6, 8):
        kp = -(-4096 // S); kp = -(-kp // 64) * 64
        bounds = [(i * kp, min(4096, (i + 1) * kp)) for i in range(S) if i * kp < 4096]
        parts = [X.run_swap(x, router.contiguous(), a, b).t() for a, b in bounds]
        rev = parts[-1].clone()
        for p in parts[-2::-1]: rev = rev + p
        tree = parts
        while len(tree) > 1:
            tree = [tree[i] + tree[i + 1] if i + 1 < len(tree) else tree[i] for i in range(0, len(tree), 2)]
        if torch.equal(rev, full): res.append(f'S{S} reverse')
        if torch.equal(tree[0], full): res.append(f'S{S} pairwise')
    print(f'M {m:4d}: {res}', flush=True)
print('RE2_DONE')
