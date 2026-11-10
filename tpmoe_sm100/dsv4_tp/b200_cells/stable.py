"""Suite-harness table for one cell: python3 -I stable.py results/b200/<cell> TAG [--stat us|us_mean] [--proto b]
Reads <cell>/TAG_{1f,1t,2f,2t}/result.json; per M the mean over routing cases and reps of each arm (per-rank median -> max over
ranks; --stat us_mean = 20%-trimmed means), best baseline (fastest non-ours arm) and best / ours (> 1 = ours faster);
then the correctness of the check case (rank 0 view)."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

NAMES = {'ours': 'ours', 'gc': 'gc', 'mega': 'Mega', 'fimega': 'fiMega', 'fitrt': 'trt'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cell')
    ap.add_argument('tag')
    ap.add_argument('--stat', default='us')
    ap.add_argument('--proto', default='b')
    ap.add_argument('--only', default='')   # comma-separated run suffixes to keep (e.g. 2f,3f,1t,2t)
    a = ap.parse_args()
    d = Path(a.cell)
    runs = sorted(p for p in d.glob(f'{a.tag}_*/result.json'))
    if a.only:
        keep = set(a.only.split(','))
        runs = [p for p in runs if p.parent.name[len(a.tag) + 1:] in keep]
    vals = defaultdict(list)
    info, chk, kern, hosts = {}, {}, set(), set()
    for p in runs:
        r = json.load(open(p))
        hosts.add(r.get('host', '?'))
        if 'ours' in r.get('arms', {}):
            kern.add(r['arms']['ours'].get('kernel', '?'))
        for Ms, pt in r['points'].items():
            M = int(Ms)
            per = defaultdict(list)
            rr, U = [], []
            for c in pt['cases'].values():
                rr.append(c['recv_max_over_mean'])
                U.append(c['unique_experts'])
                for arm, t in c['timing'].items():
                    if isinstance(t, dict) and a.proto in t:
                        per[arm].append(t[a.proto].get(a.stat, t[a.proto]['us']))
            info[M] = (pt['T'], statistics.mean(rr), statistics.mean(U), len(pt['cases']))
            for arm, v in per.items():
                vals[(M, arm)].append(statistics.mean(v))
            ck = pt.get('check') or {}
            for arm, cc in (ck.get('correctness') or {}).items():
                chk.setdefault(arm, {})[M] = cc
    arms = [x for x in NAMES if any(k[1] == x for k in vals)]
    print(f'{d.name} {a.tag}: {len(runs)} runs ({", ".join(p.parent.name for p in runs)}), hosts {sorted(hosts)}, ours = {sorted(kern)}')
    print(f'protocol {a.proto}, stat {a.stat}: us = mean over cases and reps (reps per arm in brackets when != 2)\n')
    print('| M | T | recv max/mean | U | ' + ' | '.join(NAMES[x] for x in arms) + ' | best baseline | best / ours |')
    print('|' + '---|' * (6 + len(arms)))
    for M in sorted(info):
        T, rr, U, nc = info[M]
        cells, base = [], []
        for x in arms:
            v = vals.get((M, x))
            if not v:
                cells.append('-'); continue
            m = statistics.mean(v)
            cells.append(f'{m:.1f}' + ('' if len(v) == 2 else f' [{len(v)}]'))
            if x != 'ours':
                base.append((m, NAMES[x]))
        o = vals.get((M, 'ours'))
        bb = min(base) if base else None
        ratio = f'{bb[0] / statistics.mean(o):.3f}' if (bb and o) else '-'
        print(f'| {M} | {T} | {rr:.2f} | {U:.0f} | ' + ' | '.join(cells) + f' | {bb[1] + " " + format(bb[0], ".1f") if bb else "-"} | {ratio} |')
    print('\ncorrectness (check case, real x):')
    for arm, per in chk.items():
        ks = sorted(per)
        def rng(key, fmt='.3g'):
            v = [per[k].get(key) for k in ks if isinstance(per[k].get(key), (int, float)) and not isinstance(per[k].get(key), bool)]
            return f'{min(v):{fmt}}..{max(v):{fmt}}' if v else '-'
        flags = []
        for key in ('det', 'ranks_same', 'bitwise_exact'):
            v = [per[k].get(key) for k in ks if key in per[k]]
            if v:
                flags.append(f'{key} {"all" if all(v) else "NOT all"}')
        nf = sum(int(per[k].get('nonfinite', 0) or 0) for k in ks)
        print(f'- {NAMES.get(arm, arm)}: rel_q {rng("rel_q")}, cos_q {rng("cos_q", ".5f")}, rel_exact {rng("rel_exact")}, cos_dense {rng("cos_dense", ".5f")}, '
              f'nonfinite {nf}, ' + ', '.join(flags))


if __name__ == '__main__':
    main()
