"""H200 DSV4.1-Flash TP4: tpv41.cu = port_v41_tp4.py(tp8v41.cu) (H20-TP8 agent's port_v41.py of h123's v78h21, generalized to NDEV 4:
per-rank intermediate 576 = 4 items + a 64-column tail item, NKT2 5, reduce-scatter owner (rtile + t) % 4), TP2_MODEL 1, NDEV 4.
Overrides: --opt src=..,flags=.. / dispatch=..; --ours-so."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h2cell  # noqa: E402

ADAPTER_API = 1


def make_cell(ctx):
    # FC2 tail k-tile: plain 4-slice k-step (TP2_FC2_T1S 0) by default, the 2-slice pipelined KPipeT2 (T1S 2) at M <= 32 (rank-local
    # -1.5..-2 us there, +3..6 us at M >= 128); outputs bitwise identical
    return h2cell.make(ctx, 'tpv41', 'dsv41f', [(0, 1 << 30, 'tpv41.cu', ('-DTP2_FC2_T1S=0',)), (0, 32, 'tpv41.cu', ())],
                       dict(kernel='tpv41 (port_v41_tp4 of tp8v41 @ v41_h20 7076d03 = H20 v41 port of h123 v78h21), routed-only, TP4'))
