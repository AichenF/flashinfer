"""patch_ndev.py: v77 tpmoe_b2.cu (after patch_ro.py) -> TP size as a compile-time knob TPB2_NDEV (default 4 = v77 unchanged; 8 = DSV4-Flash
TP8: INTER = 2048 / 8 = 256 -> 4 FC1 tiles per panel, 2 FC2 k-chunks (one stage per FC2 row tile), LRT 2, one-shot slots MAXM / 8)."""
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'v77' / 'tpmoe_b2.cu')
s = P.read_text()


def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, f'expected {count}, found {c}: {old[:200]}'
    s = s.replace(old, new)


rep('constexpr int NDEV = 4;\n',
    '#ifndef TPB2_NDEV\n#define TPB2_NDEV 4   // (suite_b200) TP size: 4 (v77) or 8 (DSV4-Flash TP8: INTER 256)\n#endif\n'
    'constexpr int NDEV = TPB2_NDEV;\nstatic_assert(NDEV == 4 || NDEV == 8, "TP4 / TP8");\n')
rep('"TPB2_SPEC=2: TP4, top-6, routed only (ids / weights given)"', '"TPB2_SPEC=2: TP size NDEV, top-6, routed only (ids / weights given)"')
P.write_text(s)
print('patched', P)
