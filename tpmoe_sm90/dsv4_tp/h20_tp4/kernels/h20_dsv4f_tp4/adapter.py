"""v94ex at NDEV 4 on H20 (TP4 DeepSeek-V4-Flash, routed-only ns build): suite_h20b src/tp4moe2_v94ex.cu (= tpmoe_h100 5c3aa17
tp4moe2_v94ex.cu: v94p11dev + LL epoch-parity double buffers + TP2_SH_FIRST 0 + TP2_SMALL_MAXMT 128; bitwise == v28 / served v27 ns on
all 4 ranks), built -DTP2_SHARED=0 (NDEV 4 default) with the package flags; host = src/tp4moe2_par.py (RS / AG x2 parity halves);
Humming weights via the suite's humming_prep.  --ours-so SO: a prebuilt ns .so; --ours-opt src=FILE.cu: another source in src/."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / '_h20b' / 'src'   # (tpmoe_suite import: suite_h20b/src @ ca18afa copied to kernels/_h20b/src)
SUITE = Path(os.environ.get('TPMOE_SUITE', '/home/scratch.aichenf_wwfo/tpmoe_suite'))
sys.path.insert(0, str(SUITE / 'harness'))
import sm90_tp4moe2 as F  # noqa: E402

ADAPTER_API = 1


def make_cell(ctx):
    src = ctx.opts.get('src', 'tp4moe2_v94ex.cu')
    if ctx.so:
        ext = F.load_so(ctx.so[0])
        so = F.so_info([ctx.so[0]])
    else:
        ext = F.build_shared(ctx, SRC, src, 'h20b_' + Path(src).stem + '_ns_tp4', ['-DTP2_SHARED=0'])
        so = F.so_info([ext.__file__])
    prov = dict(kernel=f'{src} (v94ex lineage, exact == v28), ns routed-only, TP4', so=so)
    return F.Tp4moe2Cell(ctx, Path(src).stem, SRC / 'tp4moe2_par.py', [(0, 1 << 30, ext)], prov)
