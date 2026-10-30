import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); a = ap.parse_args()
t = time.time()
import tpb2
E = tpb2.load_ext()
print('BUILT', E.__file__, 'smem', E.smem_alloc(), f'{time.time() - t:.0f}s', flush=True)
