"""h123 (H20 TP8 DSV4-Flash, routed-only ns builds), M-dispatched as served: v78h21 default, v94h13 at M 64..127, v94h21 at M 128..341.
Default (since the SM90 back-to-back race fix, 2026-11-09): BUILT from src/tp4moe2_ns.{v78h21,v94h13,v94h21}.cu = H20-TP8's fixed copies
(v41_h20/fix_b2b/h123: the served sources + the TP2_B2B_FIX barrier) with the served build's flags (-O3 sm_90a -DTP2_SHARED=0 -DTP2_NDEV=8,
recovered from the served build.ninja).  --opt prebuilt=1: the three served-identical prebuilt .so's in so/ (PRE-FIX).  --ours-so overrides."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'harness'))
import sm90_tp4moe2 as F  # noqa: E402

ADAPTER_API = 1
RANGES = [(0, 1 << 30, 'v78h21'), (64, 127, 'v94h13'), (128, 341, 'v94h21')]


def make_cell(ctx):
    if ctx.so:
        sos = []
        for s in ctx.so:
            p, _, rng = s.partition('@')
            lo, hi = (int(v) for v in rng.split('-')) if rng else (0, 1 << 30)
            sos.append((lo, hi, p))
        exts = [(lo, hi, F.load_so(p)) for lo, hi, p in sos]
        prov = dict(kernel='h123 (--ours-so)', so=F.so_info([p for _, _, p in sos]))
    elif ctx.opts.get('prebuilt', '0') == '1':
        sos = [(lo, hi, str(HERE / 'so' / f'tpmoe_var_ns_{t}_tp8.so')) for lo, hi, t in RANGES]
        exts = [(lo, hi, F.load_so(p)) for lo, hi, p in sos]
        prov = dict(kernel='h123 served prebuilt .so (PRE-FIX)', so=F.so_info([p for _, _, p in sos]))
    else:
        exts = [(lo, hi, F.build_shared(ctx, HERE / 'src', f'tp4moe2_ns.{t}.cu', f'h123fix_{t}_ns_tp8', ['-DTP2_SHARED=0', '-DTP2_NDEV=8']))
                for lo, hi, t in RANGES]
        prov = dict(kernel='h123 (v78h21 + v94h13@64-127 + v94h21@128-341) built from the TP2_B2B_FIX sources, ns routed-only, TP8',
                    so=F.so_info([e.__file__ for _, _, e in exts]))
    prov['dispatch'] = [(lo, hi, t) for lo, hi, t in RANGES]
    return F.Tp4moe2Cell(ctx, 'h123', HERE / 'tp4moe2.py', exts, prov)
