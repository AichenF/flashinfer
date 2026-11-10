"""tp8v41h2/mk_g.py [SRC DST]: tp8v41.cu (tp8_v41_b300 @d91f622: H2 / F2H behind TPB2_F2H, LLS + tail knobs already in) -> tp8v41g.cu + the suite_b200
LLS guard (trap if gridDim != TPB2_GRID: one reader per LL slot)."""
from pathlib import Path
import sys
H = Path(__file__).resolve().parent
SRC, DST = (sys.argv[1:3] if len(sys.argv) > 2 else ('tp8v41.cu', 'tp8v41g.cu'))
s = (H / SRC).read_text()
a = 'void lls_allreduce(const Params& P, int b, int G, int M, int tid, unsigned epoch) {\n'
assert s.count(a) == 1
s = s.replace(a, a + '    if (TPB2_GRID && gridDim.x != static_cast<unsigned>(TPB2_GRID)) __trap();   // (suite_b200 LLS guard: one reader per slot)\n')
(H / DST).write_text(s)
print(f'wrote {DST} from {SRC}')
