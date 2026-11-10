"""H200 DSV4-Flash TP4: v94ex (H100 TP4 package kernel, suite_h200/src/tp4moe2_v94ex.cu == pkg serving_tpmoe_tp4_h100 tp4moe2_ns.cu,
sha256 bba973fb..), routed-only ns build (NDEV 4, -DTP2_SHARED=0).  Overrides: --opt src=..,flags=A=1+B=2 / dispatch=..; --ours-so."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h2cell  # noqa: E402

ADAPTER_API = 1


def make_cell(ctx):
    return h2cell.make(ctx, 'v94ex', 'dsv4f', [(0, 1 << 30, 'tp4moe2_v94ex.cu', ())],
                       dict(kernel='v94ex (tpmoe_h100 pkg ca7302a = github AichenF/flashinfer sm90/tpmoe-h100 86d75ac), ns routed-only, TP4'))
