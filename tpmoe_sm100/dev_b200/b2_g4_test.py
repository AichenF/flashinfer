"""gather4 semantics on B200 (b2_g4_test.cu).  usage: python3 b2_g4_test.py --model x --cap x --out x"""
import argparse, os
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.parse_args()
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0a'
    (SRV / 'build_g4').mkdir(exist_ok=True)
    X = load(name='b2_g4_test', sources=[str(SRV / 'b2_g4_test.cu')], build_directory=str(SRV / 'build_g4'),
             extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a'], extra_ldflags=['-lcuda'], verbose=False)
    r = torch.arange(64, dtype=torch.int32).view(64, 1); col = torch.arange(4096, dtype=torch.int32).view(1, 4096)
    x = ((r * 7 + col * 13) % 251).to(torch.uint8).cuda()
    s = ((torch.arange(64).view(64, 1) * 3 + torch.arange(256).view(1, 256) * 5) % 253).to(torch.uint8).cuda()
    rows = [5, 9, 2, 40]
    for o1, o2 in ((0, 0), (512, 0), (128, 0), (0, 64), (0, 16), (0, 128)):
        try:
            o = X.run(x, s, 3, 1, o1, o2).cpu()
        except Exception as e:
            print(f'o1 {o1} o2 {o2}: ERROR {str(e)[:80]}', flush=True)
            X = load(name='b2_g4_test', sources=[str(SRV / 'b2_g4_test.cu')], build_directory=str(SRV / 'build_g4'),
                     extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a'], extra_ldflags=['-lcuda'], verbose=False)
            continue
        ok = []
        for i, rr in enumerate(rows):
            want = x[rr, 3 * 128:4 * 128].cpu()
            got = torch.empty(128, dtype=torch.uint8)
            base = o1 + i * 128
            for g in range(8):
                q = (base >> 7) & 7
                got[g * 16:(g + 1) * 16] = o[base + ((g ^ q) * 16):base + ((g ^ q) * 16) + 16]
            ok.append(bool(torch.equal(got, want)))
        sok = [bool(torch.equal(o[2048 + o2 + 16 * i:2048 + o2 + 16 * (i + 1)], s[rr, 48:64].cpu())) for i, rr in enumerate(rows)]
        print(f'o1 {o1} o2 {o2}: x ok {ok}  sf ok {sok}', flush=True)
    print('G4_DONE')
if __name__ == '__main__':
    main()
