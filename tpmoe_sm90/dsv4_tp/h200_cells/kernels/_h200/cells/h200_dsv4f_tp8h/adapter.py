"""H200 DSV4-Flash on 8 GPUs, TP-family HYBRID variant (not the pure TP8 cell): tokens replicated on every rank, in-kernel 8-way
all-reduce, weights split as 2 expert groups x TP4 slices (rank r: experts [128 (r / 4), +128), TP4 slice r % 4) -- tpv41h.cu at
TP2_MODEL 0 (= h123 code) -DTP2_NDEV=8 -DTP2_ISPLIT=4 -DTP2_FC2_WIDE=0; per group its output is bitwise the v94ex TP4 kernel's on the
group's experts (rank-local check).  Dispatch: h123 (v78h21 / v94h13 / v94h21) at M <= 255, hybrid at M >= 256 (8 GPUs, real trace: equal at M32..128, -2..-6 us at M256..512)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h2cell  # noqa: E402

ADAPTER_API = 1


def make_cell(ctx):
    return h2cell.make(ctx, 'h123-hyb', 'dsv4f', [(0, 1 << 30, 'tpv41h.cu', ('-DTP2_ISPLIT=4', '-DTP2_FC2_WIDE=0')),
                                                  (0, 63, 'tp4moe2_v78p12dev.cu', ()), (64, 127, 'tp4moe2_v94p10dev.cu', ()), (128, 255, 'tp4moe2_v94p11dev.cu', ())],
                       dict(kernel='hybrid (8-way AR, 2 expert groups x TP4 slices; tpv41h MODEL 0) at M >= 256, h123 below'))
