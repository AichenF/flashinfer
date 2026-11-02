"""Analyze tpmoe_b2 event traces (TPB2_TRACE=1; b2_layer_ab.py --trace saves trace_m{M}_r{rank}.pt).
Per CTA: clock64 events mapped to ns by the start / end globaltimer anchors; times are us from the earliest CTA start of the rank.
usage: python3 b2_trace_an.py trace_m1_r0.pt [..] [--cta N ..] [--full]"""
import argparse
import statistics as stt

NAMES = {1: 'start', 2: 'end', 3: 'pdl', 4: 'gate_w', 5: 'quant_w', 6: 'tables', 7: 'roles_done', 8: 'sh_early', 10: 'push', 11: 'prod_fc1_done',
         12: 'prod_hwait', 20: 'mma_acc', 21: 'mma_commit', 22: 'mma_data', 30: 'b_h', 31: 'b_x', 40: 'epi_acc', 41: 'epi_part',
         42: 'merge_wait', 43: 'merge_done', 44: 'h_pub', 45: 'fc2_store', 46: 'epi_done', 47: 'merge_flags', 48: 'part_fence', 55: 'h_written', 49: 't1_item', 50: 't1_rt_ok', 53: 't1_done',
         54: 'ar_done'}


def dec(arg):
    x = (arg >> 44) & 0xF
    if x == 0xF:
        x = -1
    return dict(x=x, y=(arg >> 24) & 0xFFFF, z=(arg >> 12) & 0xFFF, w=arg & 0xFFF, a=(arg >> 40) & 0xF)


def load(path):
    import torch
    d = torch.load(path)
    ev, G, TRN = d['ev'], d['grid'], d['trn']
    tr = ev[G * 32:].view(G, TRN, 2).tolist()
    ctas = []
    for c in range(G):
        rows = []
        for clk, w in tr[c]:
            w &= (1 << 64) - 1
            code = w >> 56
            if code == 0:
                continue
            rows.append((clk & ((1 << 64) - 1), code, (w >> 48) & 0xFF, w & 0xFFFFFFFFFFFF))
        ctas.append(rows)
    # anchors -> ns
    base = None
    out = []
    for c, rows in enumerate(ctas):
        st = [r for r in rows if r[1] == 1]
        en = [r for r in rows if r[1] == 2]
        if not st or not en:
            out.append([])
            continue
        c1, g1 = st[0][0], st[0][3]
        c2, g2 = en[0][0], en[0][3]
        k = (g2 - g1) / max(1, c2 - c1)
        evs = [(g1 + (r[0] - c1) * k, r[1], r[2], r[3]) for r in rows]
        base = g1 if base is None else min(base, g1)
        out.append(evs)
    res = []
    for evs in out:
        res.append(sorted([((t - base) / 1e3, code, warp, arg) for t, code, warp, arg in evs]))
    return dict(m=d['m'], rank=d['rank'], ctas=res, G=G)


def q(v):
    if not v:
        return '-'
    v = sorted(v)
    return f'{stt.median(v):6.2f}/{v[-1]:6.2f}'


def analyze(path, show=(), full=False):
    D = load(path)
    C = D['ctas']
    print(f'== {path}: M {D["m"]} rank {D["rank"]} grid {D["G"]}')
    first = lambda evs, code: next((t for t, c, w, a in evs if c == code), None)
    last = lambda evs, code: next((t for t, c, w, a in reversed(evs) if c == code), None)
    # prologue
    ph = {'pdl': [], 'gate(max warp)': [], 'quant(max warp)': [], 'tables': [], 'first_push': [], 'mma_data1': [], 'prod_fc1_done': [],
          'epi_done': [], 'roles_done': [], 't1_done': [], 'end': []}
    for evs in C:
        if not evs:
            continue
        ph['pdl'].append(first(evs, 3))
        g = [t for t, c, w, a in evs if c == 4]
        if g: ph['gate(max warp)'].append(max(g))
        g = [t for t, c, w, a in evs if c == 5]
        if g: ph['quant(max warp)'].append(max(g))
        for nm, code in (('tables', 6), ('first_push', 10), ('mma_data1', 22), ('prod_fc1_done', 11), ('epi_done', 46), ('roles_done', 7),
                         ('t1_done', 53), ('end', 2)):
            t = first(evs, code)
            if t is not None: ph[nm].append(t)
    print('  phases (median/max us): ' + '  '.join(f'{k} {q([x for x in v if x is not None])}' for k, v in ph.items()))
    # FC1 pieces: push -> mma_acc -> data -> commit -> epi_acc -> (part | merge_wait -> merge_done -> h_pub)
    iv = {k: [] for k in ('push->mma_acc', 'mma_acc->data', 'data->commit', 'commit->epi_acc', 'epi_acc->part', 'epi_acc->merge_wait',
                          'merge_wait->merge_done', 'merge_done->h_pub', 'FC2 push->b_h', 'FC2 push->commit', 'FC2 commit->store',
                          'FC2 prod hwait')}
    hpub, panel_last = [], {}
    fc2_store_last = 0.0
    for evs in C:
        pend = {}   # role-ordered queues of pieces
        pushes = [(t, dec(a)) for t, c, w, a in evs if c == 10]
        accs = [(t, dec(a)) for t, c, w, a in evs if c == 20]
        datas = [t for t, c, w, a in evs if c == 22]
        coms = [(t, dec(a)) for t, c, w, a in evs if c == 21]
        eacc = [(t, dec(a)) for t, c, w, a in evs if c == 40]
        parts = [t for t, c, w, a in evs if c == 41]
        mw = [t for t, c, w, a in evs if c == 42]
        md = [t for t, c, w, a in evs if c == 43]
        hp = [(t, a) for t, c, w, a in evs if c == 44]
        for t0 in [t for t, c, w, a in evs if c == 42]:
            t1 = next((t for t, c, w, a in evs if c == 47 and t >= t0), None)
            if t1 is not None: iv.setdefault('merge_wait->flags', []).append(t1 - t0)
        for t0 in [t for t, c, w, a in evs if c == 48]:
            t1 = next((t for t, c, w, a in evs if c == 41 and t >= t0), None)
            if t1 is not None: iv.setdefault('part_fence->flag', []).append(t1 - t0)
        for t0 in [t for t, c, w, a in evs if c == 55]:
            t1 = next((t for t, c, w, a in evs if c == 44 and t >= t0), None)
            if t1 is not None: iv.setdefault('h_written->h_pub', []).append(t1 - t0)
        bh = [(t, dec(a)) for t, c, w, a in evs if c == 30]
        st = [(t, dec(a)) for t, c, w, a in evs if c == 45]
        hw = [(t, a) for t, c, w, a in evs if c == 12]
        # FC1: pieces in push order (kinds 1, 3), one accumulator each
        p1 = [p for p in pushes if p[1]['x'] in (1, 3)]
        a1 = [p for p in accs if p[1]['x'] in (1, 3)]
        c1 = [p for p in coms if p[1]['x'] in (1, 3)]
        e1 = [p for p in eacc if p[1]['x'] in (1, 3)]
        for i in range(min(len(p1), len(a1), len(c1), len(e1))):
            iv['push->mma_acc'].append(a1[i][0] - p1[i][0])
            if i < len(datas):
                iv['mma_acc->data'].append(datas[i] - a1[i][0])
                iv['data->commit'].append(c1[i][0] - datas[i])
            iv['commit->epi_acc'].append(e1[i][0] - c1[i][0])
        # (contributor partial / owner merge per FC1 piece, in order)
        for t in parts:
            prev = max([e[0] for e in e1 if e[0] <= t], default=None)
            if prev is not None: iv['epi_acc->part'].append(t - prev)
        for t0, t1 in zip(mw, md):
            prev = max([e[0] for e in e1 if e[0] <= t0], default=None)
            if prev is not None: iv['epi_acc->merge_wait'].append(t0 - prev)
            iv['merge_wait->merge_done'].append(t1 - t0)
        for t, a in hp:
            prev = max([x for x in md if x <= t], default=None)
            if prev is None:
                prev = max([e[0] for e in e1 if e[0] <= t], default=None)
            if prev is not None: iv['merge_done->h_pub'].append(t - prev)
            hpub.append(t)
            if (a >> 20) & 1:
                panel_last[a & 0xFFFFF] = t
        # FC2 units
        p2 = [p for p in pushes if p[1]['x'] in (2, 4)]
        for t, d in p2:
            b = next((x for x in bh if x[1]['y'] == d['y'] and x[1]['x'] == d['x'] and x[0] >= t), None)
            if b: iv['FC2 push->b_h'].append(b[0] - t)
            cm = [x for x in coms if x[1]['y'] == d['y'] and x[1]['x'] == d['x'] and x[0] >= t]
            if cm: iv['FC2 push->commit'].append(cm[-1][0] - t)
            for x in cm:
                s2 = next((y for y in st if y[1]['y'] == d['y'] and y[1]['a'] == x[1]['a'] and y[0] >= x[0]), None)
                if s2: iv['FC2 commit->store'].append(s2[0] - x[0])
        for t, u in hw:
            nxt = next((x[0] for x in p2 if x[0] >= t), None)
            if nxt is not None: iv['FC2 prod hwait'].append(nxt - t)
        if st:
            fc2_store_last = max(fc2_store_last, st[-1][0])
    for k, v in iv.items():
        if v:
            print(f'  {k:24s} n {len(v):5d}  median/max {q(v)}  sum {sum(v):8.1f}')
    if panel_last:
        v = sorted(panel_last.values())
        print(f'  panels ready (last FC1 item h published): n {len(v)}  first {v[0]:.2f}  median {stt.median(v):.2f}  last {v[-1]:.2f}')
    print(f'  last FC2 row tile stored {fc2_store_last:.2f}')
    # per CTA: units / pieces counts and the last few CTAs to finish their roles
    fin = sorted(((first(evs, 7) or 0.0, c) for c, evs in enumerate(C) if evs), reverse=True)
    print('  last CTAs to finish roles: ' + '  '.join(f'cta {c} {t:.2f}' for t, c in fin[:6]))
    t1 = sorted(((first(evs, 50) or 0.0, c, first(evs, 49) or 0.0) for c, evs in enumerate(C) if evs and first(evs, 50) is not None))
    if t1:
        print('  T1 rt wait: ' + '  '.join(f'cta {c} {a:.2f}->{t:.2f}' for t, c, a in t1[-4:]))
    # the latest FC1 item completions (h published) with their owner's timeline, the latest FC2 row tiles stored
    rows = []
    for c, evs in enumerate(C):
        for t, code, w, a in evs:
            if code == 44:
                ea = max([x[0] for x in evs if x[1] == 40 and x[0] <= t], default=-1)
                mw = max([x[0] for x in evs if x[1] == 42 and x[0] <= t], default=-1)
                md = max([x[0] for x in evs if x[1] == 43 and x[0] <= t], default=-1)
                pu = max([(x[0], dec(x[3])) for x in evs if x[1] == 10 and x[0] <= t and dec(x[3])['x'] in (1, 3)], default=(-1, {}), key=lambda z: z[0])
                rows.append((t, c, a & 0xFFFFF, (a >> 20) & 1, pu[0], pu[1], ea, mw, md))
    rows.sort()
    print('  latest h publishes (t, cta, panel, last, push t / desc, epi_acc, merge_wait, merge_done):')
    for r in rows[-10:]:
        d = r[5]
        print(f'     {r[0]:7.2f} cta {r[1]:3d} p {r[2]:3d} last {r[3]}  push {r[4]:6.2f} x{d.get("x")} y{d.get("y")} z{d.get("z")} w{d.get("w")}  '
              f'acc {r[6]:6.2f}  mw {r[7]:6.2f}  md {r[8]:6.2f}')
    rows = []
    for c, evs in enumerate(C):
        for t, code, w, a in evs:
            if code == 45:
                d = dec(a)
                pu = next((x[0] for x in evs if x[1] == 10 and dec(x[3])['y'] == d['y'] and dec(x[3])['x'] == d['x']), -1)
                cm = next((x[0] for x in evs if x[1] == 21 and dec(x[3])['y'] == d['y'] and dec(x[3])['x'] == d['x'] and dec(x[3])['a'] == d['a']), -1)
                hw = max([x[0] for x in evs if x[1] == 12 and x[0] <= pu], default=-1)
                rows.append((t, c, d['x'], d['y'], d['a'], hw, pu, cm))
    rows.sort()
    print('  latest FC2 stores (t, cta, kind, unit, a, hwait start, push, commit):')
    for r in rows[-8:]:
        print(f'     {r[0]:7.2f} cta {r[1]:3d} x{r[2]} u{r[3]:4d} a{r[4]}  hwait {r[5]:6.2f}  push {r[6]:6.2f}  commit {r[7]:6.2f}')
    for c in list(show) + ([fin[0][1]] if full else []):
        print(f'  -- CTA {c} events:')
        for t, code, w, a in C[c]:
            ds = dec(a)
            extra = f'x{ds["x"]} y{ds["y"]} z{ds["z"]} w{ds["w"]} a{ds["a"]}' if code in (10, 20, 21, 30, 31, 40, 41, 43, 45) else f'{a if code not in (1, 2) else ""}'
            print(f'     {t:8.2f}  w{w:2d}  {NAMES.get(code, code):14s} {extra}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--cta', type=int, nargs='*', default=[])
    ap.add_argument('--full', action='store_true')
    a = ap.parse_args()
    for p in a.paths:
        analyze(p, a.cta, a.full)


if __name__ == '__main__':
    main()
