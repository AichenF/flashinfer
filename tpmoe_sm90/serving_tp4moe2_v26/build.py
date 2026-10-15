"""Prebuild both serving extensions with the hook's exact load() calls (serving image; no GPU needed)."""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp4_ext import load_ext
for kind in ['sh', 'ns']:
    E = load_ext(kind)
    print('BUILT', kind, E.__file__, hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(), 'MAXM', E.MAXM, flush=True)
