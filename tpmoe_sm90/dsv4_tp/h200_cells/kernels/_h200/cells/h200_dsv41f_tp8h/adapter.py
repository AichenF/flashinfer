"""H200 DSV4.1-Flash on 8 GPUs, TP-family HYBRID variant (not the pure TP8 cell): all M tokens replicated on every rank, in-kernel 8-way
all-reduce as TP8, but the weights are split as 2 expert groups x TP4 slices (rank r: experts [192 (r / 4), 192 (r / 4) + 192), TP4
intermediate slice r % 4) -- tpv41h.cu = port_isplit.py(tpv41.cu) built -DTP2_NDEV=8 -DTP2_ISPLIT=4 (ids of the other group are skipped
in-kernel).  Fatter per-rank slices (FC1 4 items + tail, FC2 K 576) amortize the per-item / per-unit overheads; costs the group imbalance.
Default dispatch: pure TP8 (tpv41.cu) at M <= 8 (tie), hybrid at M >= 9 (8 GPUs, real trace: M16 -4.0, M32 -4.4, M64.. -9..-31 us).  Overrides: --opt dispatch=..; --ours-so."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h2cell  # noqa: E402

ADAPTER_API = 1


def make_cell(ctx):
    return h2cell.make(ctx, 'tpv41-hyb', 'dsv41f', [(0, 1 << 30, 'tpv41h.cu', ('-DTP2_ISPLIT=4', '-DTP2_FC2_T1S=0')), (0, 8, 'tpv41.cu', ('-DTP2_TAIL4=0',))],
                       dict(kernel='tpv41h hybrid (8-way AR, 2 expert groups x TP4 slices) for M > 8, pure TP8 tpv41 for M <= 8'))
