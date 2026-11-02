"""SASS size of tpb2_kernel per build: --sets=flags+flags+... (comma-separated -D flags per build; tags as the harness' arm KS2 (_b))."""
import sys, subprocess, re, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpb2
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.add_argument('--sets', required=True); ap.add_argument('--suffix', default='_b')
a = ap.parse_args()
tagf = lambda f: ('_' + f.replace('-D', '').replace('=', '').replace(' ', '_')) if f else ''
sets = [s.replace(',', ' ').strip() for s in a.sets.split('+')]
def one(f):
    tag = tagf(f) + a.suffix
    tpb2.load_ext(tag=tag, flags=tuple(f.split()))
    import re as _re
    o = tpb2.OUT / ('build_tpb2' + _re.sub(r'[^A-Za-z0-9_]', '_', tag)) / 'tpmoe_b2.cuda.o'
    sass = subprocess.run(['cuobjdump', '-sass', str(o)], capture_output=True, text=True).stdout
    out = {}
    for fn in re.split(r'\n\s*Function : ', sass)[1:]:
        name = fn.split('\n', 1)[0].strip()
        n = len(re.findall(r'/\*[0-9a-f]{4,}\*/\s+[^;]*;', fn))
        out['tpb2_kernel' if 'tpb2_kernel' in name else name[:40]] = n
    return f, out
with ThreadPoolExecutor(len(sets)) as ex:
    for f, out in ex.map(one, sets):
        print(f'SASS [{f}]: ' + '  '.join(f'{k} {v}' for k, v in out.items() if 'tpb2' in k), flush=True)
