"""SASS instructions of tpb2_kernel per source line (-lineinfo build): --flags=a,b,..  -> runs/<out>.lines (line count) + top lines."""
import sys, subprocess, re, argparse, os, tempfile, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpb2
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--cap'); ap.add_argument('--out'); ap.add_argument('--flags', default='')
a = ap.parse_args()
f = a.flags.replace(',', ' ').strip() + ' -lineinfo'
tag = ('_' + f.replace('-D', '').replace('=', '').replace(' ', '_').replace('-', '')) + '_L'
tpb2.load_ext(tag=tag, flags=tuple(f.split()))
o = tpb2.OUT / ('build_tpb2' + re.sub(r'[^A-Za-z0-9_]', '_', tag)) / 'tpmoe_b2.cuda.o'
d = tempfile.mkdtemp()
subprocess.run(['cuobjdump', '-xelf', 'all', str(o)], cwd=d, check=True, capture_output=True)
cub = [x for x in os.listdir(d) if x.endswith('.cubin')][0]
txt = subprocess.run(['nvdisasm', '-gi', '-c', os.path.join(d, cub)], capture_output=True, text=True).stdout
open(a.out + '.gi.txt', 'w').write(txt[:400000])
cnt, cur, infn = collections.Counter(), -1, False
for l in txt.split('\n'):
    if '.text.' in l and ':' in l and l.lstrip().startswith('.text'):
        infn = 'tpb2_kernel' in l
    if '//## File' in l:
        ls = re.findall(r'tpmoe_b2\.cu", line (\d+)', l)
        cur = int(ls[-1]) if ls else -1   # (outermost: the line in the kernel body that inlined it)
        continue
    if re.search(r'//## File', l): cur = -1; continue
    if infn and re.match(r'\s+/\*[0-9a-f]{4,}\*/\s+\S', l): cnt[cur] += 1
tot = sum(cnt.values())
with open(a.out + '.lines', 'w') as fo:
    for k in sorted(cnt): fo.write(f'{k} {cnt[k]}\n')
print('TOTAL', tot)
for k, v in cnt.most_common(40): print('LINE', k, v)
