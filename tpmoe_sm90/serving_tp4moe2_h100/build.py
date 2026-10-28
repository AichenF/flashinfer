"""Prebuild the serving extensions with the hook's exact load() calls (serving image; no GPU needed): sh + ns for TP4 and TP8."""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp4_ext import load_ext
for kind, ndev in [('sh', 4), ('ns', 4), ('sh', 8), ('ns', 8)]:
    if sys.argv[1:] and str(ndev) not in sys.argv[1:]:   # build.py [4] [8]: only these TP sizes
        continue
    E = load_ext(kind, ndev)
    print('BUILT', kind, ndev, E.__file__, hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(), 'MAXM', E.MAXM, flush=True)
