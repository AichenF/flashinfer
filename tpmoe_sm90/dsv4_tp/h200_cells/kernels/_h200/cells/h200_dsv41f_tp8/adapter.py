"""H200 DSV4.1-Flash TP8: tpv41.cu at NDEV 8 (SASS-equivalent to the H20-TP8 agent's tp8v41.cu routed-only build: outputs bitwise equal),
TP2_MODEL 1.  Overrides: --opt src=..,flags=.. / dispatch=..; --ours-so."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h2cell  # noqa: E402

ADAPTER_API = 1


def make_cell(ctx):
    # T4 split tail off at M <= 16 (8 GPUs real trace, 2 reps: M8 -1.2, M16 -1.1 us; outputs bitwise identical)
    return h2cell.make(ctx, 'tpv41', 'dsv41f', [(0, 1 << 30, 'tpv41.cu', ()), (0, 16, 'tpv41.cu', ('-DTP2_TAIL4=0',))],
                       dict(kernel='tpv41 (= tp8v41 @ v41_h20 7076d03 incl. KPipe1 FC2 tail, H20 v41 port of h123 v78h21), routed-only, TP8'))
