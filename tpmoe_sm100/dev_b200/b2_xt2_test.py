"""CPU unit test of xtrace.decode_v2 (XTrace 8-B layout): a valid warp, an unpaired end, an overflowed warp and a gap."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import torch
import xtrace as X
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.parse_args()
G, W, D = 2, X.WARPS, X.DEPTH
b = torch.zeros(X.words(G, 2), dtype=torch.int64)
hdr = b[:G * X.HDR].view(G, X.HDR); hdr[0, 4] = 7; hdr[1, 4] = 9
slab = b[G * X.HDR:].view(G, W, X.SLAB_HDR + D)
def head(c, w, sm):
    h = slab[c, w, :X.SLAB_HDR].view(torch.int32); h[0] = 0x58543202; h[1] = sm; h[2] = c; h[3] = w
def rec(c, w, i, clk, site, kind, a0=0):
    pay = (((a0 >> 28) & 7) << 20) | (a0 & 0xFFFFF)
    slab[c, w, X.SLAB_HDR + i] = (((site | (kind << 7) | (pay << 9)) & 0xFFFFFFFF) << 32) | (clk & 0xFFFFFFFF)
head(0, 1, 7); rec(0, 1, 0, 100, 22, 1); rec(0, 1, 1, 150, 22, 2); rec(0, 1, 2, 160, 21, 1, (1 << 28) | 32); rec(0, 1, 3, 900, 21, 2)
head(0, 2, 7); rec(0, 2, 0, 50, 11, 2)                                   # unpaired end
head(1, 0, 9); [rec(1, 0, i, 10 * i + 1, 47, 0) for i in range(D)]; slab[1, 0, :X.SLAB_HDR].view(torch.int32)[4] = 1   # overflow
head(1, 3, 9); rec(1, 3, 0, 5, 51, 1); rec(1, 3, 2, 9, 51, 2)            # gap (noncontiguous)
_, recs, rep = X.decode_v2(b, G)
print(rep)
assert rep['overflow_warps'] == 1 and rep['unpaired_end'] == 1 and rep['noncontiguous_warps'] == 1 and not rep['valid']
assert recs[0][1][2] == (160, 21 | (1 << 12), (1 << 28) | 32), recs[0][1][2]
print('XT2_TEST_OK')
