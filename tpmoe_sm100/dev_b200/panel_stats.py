"""panel counts per M at NMAX 32 vs 64 from the captured x and the production router + gate (rank-independent)."""
import argparse, sys
from pathlib import Path
import torch
SRV = Path(__file__).resolve().parent
sys.path.insert(0, str(SRV))
import b2_common as C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out')
    args = ap.parse_args()
    dev = torch.device('cuda', 0)
    wm = C.weight_map(args.model)
    router, bias = C.load_router(args.model, wm, 20, dev)
    ms = [16, 24, 32, 48, 64, 96, 128]
    xs = C.capture_pool(args.cap, ms, dev)
    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
    for m in ms:
        lg = torch.mm(xs[m], router.t(), out_dtype=torch.float32)
        w, i = moe_fused_gate(lg, bias, topk=6, scoring_func='sqrtsoftplus', num_fused_shared_experts=0, renormalize=True,
                              routed_scaling_factor=1.5, apply_routed_scaling_factor_on_output=False)
        cnt = torch.bincount(i.flatten().long(), minlength=256)
        act = int((cnt > 0).sum())
        p32 = int(((cnt + 31) // 32).sum()); p64 = int(((cnt + 63) // 64).sum())
        print(f'M {m:3d}: active experts {act:3d}  max tokens/expert {int(cnt.max()):3d}  panels NMAX32 {p32:3d}  NMAX64 {p64:3d}  (-{100 * (p32 - p64) / p32:.1f}%)', flush=True)
    print('PANEL_STATS_DONE')


if __name__ == '__main__':
    main()
