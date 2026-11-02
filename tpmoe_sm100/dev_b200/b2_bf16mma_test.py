"""Is a cuBLAS router K-part (torch.mm fp32 out) == a tcgen05 kind::f16 K16 chain in K order? (b2_bf16mma_test.cu)  usage: ... --model M --cap C --out O"""
import argparse, os, sys
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); args = ap.parse_args()
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
(SRV / 'build_bmt').mkdir(exist_ok=True)
X = load(name='b2_bf16mma_test', sources=[str(SRV / 'b2_bf16mma_test.cu')], build_directory=str(SRV / 'build_bmt'),
         extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a'], verbose=False)
dev = torch.device('cuda', 0)
wm = C.weight_map(args.model)
router, bias = C.load_router(args.model, wm, 20, dev)
xs = C.capture_pool(args.cap, [1, 2, 4, 8, 20, 32, 64], dev)
for m, parts in ((8, [(0, 1024), (1024, 2048)]), (20, [(0, 1408), (2816, 4096)]), (32, [(0, 1024)]), (64, [(0, 1408)])):
    x = xs[m]
    for k0, k1 in parts:
        ref = torch.mm(x[:, k0:k1].contiguous(), router[:, k0:k1].contiguous().t(), out_dtype=torch.float32)
        try:
            got = X.run(x.contiguous(), router.contiguous(), k0, k1).t()
        except Exception as e:
            print(f'M {m} K [{k0},{k1}): ERROR {str(e)[:120]}', flush=True); continue
        ne = int((got != ref).sum())
        print(f'M {m:3d} K [{k0:4d},{k1:4d}): tcgen05 K16 chain == cuBLAS part: {ne == 0}  mismatches {ne} / {ref.numel()}  max|d| {float((got - ref).abs().max()):.3e}', flush=True)
for m, parts in ((8, [(0, 1024), (3072, 4096)]), (20, [(1408, 2816)]), (64, [(0, 1408), (2816, 4096)])):
    x = xs[m]
    for k0, k1 in parts:
        ref = torch.mm(x[:, k0:k1].contiguous(), router[:, k0:k1].contiguous().t(), out_dtype=torch.float32)
        got = X.run_swap(x.contiguous(), router.contiguous(), k0, k1).t()
        ne = int((got != ref).sum())
        print(f'SWAP (A = tokens) M {m:3d} K [{k0:4d},{k1:4d}): == cuBLAS part: {ne == 0}  mismatches {ne}', flush=True)
from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
for m in (1, 2, 4):
    x = xs[m]
    ref = dsv3_router_gemm(x, router, out_dtype=torch.float32)
    got = X.run_dsv3(x.contiguous(), router.contiguous())
    ne = int((got != ref).sum())
    print(f'DSV3 emulation M {m}: == dsv3_router_gemm: {ne == 0}  mismatches {ne}  max|d| {float((got - ref).abs().max()):.3e}', flush=True)
print('BMT_DONE')
