"""H200 DSV4-Flash TP8: h123 (H20 TP8 final, 3-range M dispatch) rebuilt for sm_90a from its sources: v78h21 = tp4moe2_v78p12dev.cu
(sha256 62006dc5..) default, v94h13 = tp4moe2_v94p10dev.cu (a54e3bdc..) at M 64..127, v94h21 = tp4moe2_v94p11dev.cu (84fd8f91..) at
M 128..341; ns routed-only builds (NDEV 8, -DTP2_SHARED=0).  Overrides: --opt src=..,flags=.. / dispatch=..; --ours-so."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h2cell  # noqa: E402

ADAPTER_API = 1


def make_cell(ctx):
    return h2cell.make(ctx, 'h123', 'dsv4f', [(0, 1 << 30, 'tp4moe2_v78p12dev.cu', ()), (64, 127, 'tp4moe2_v94p10dev.cu', ()),
                                              (128, 341, 'tp4moe2_v94p11dev.cu', ())],
                       dict(kernel='h123 (v78h21 + v94h13@64-127 + v94h21@128-341, H20 dev 70dbda3 sources), ns routed-only, TP8'))
