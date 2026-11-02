"""xtrace.py -- host side of xtrace.cuh (after Xtrace, arXiv 2609.28769): decode the per-warp rings, map SM cycles onto the global
ns timeline with each CTA's start / end anchors, rebuild scopes (begin / end pairs per warp) and analyze them:

  * CTA residency spread (start / end / SM), wave dispatch;
  * per-scope duration distributions, per-role wait breakdowns (time inside wait scopes);
  * tile-duration correlation within a CTA (consecutive MMA pieces);
  * the latest-finishing chains (h publish per panel, FC2 tiles, T1 / all-reduce).

  dump(buf, grid, meta, path)          (node, torch): raw buffer -> compact JSON (records already in ns)
  python3 xtrace.py show T.json        text report for agents
  python3 xtrace.py html T.json [..] -o out.html   interactive timeline panel (one or more traces, selectable)
"""
import json
import math
import statistics as stt
import sys
from pathlib import Path

HDR, WARPS, DEPTH = 8, 12, 1024
# site -> (name, role row); kinds from the id bits (0x1000 begin, 0x2000 end, else point)
SITES = {
    2: ('pdl_wait', 'pro'), 3: ('gate', 'pro'), 4: ('quant', 'pro'), 5: ('tables', 'pro'), 6: ('sh_early_tma', 'pro'), 7: ('gate_warm', 'pro'), 8: ('xflags_in', 'pro'), 64: ('tab.clr', 'pro'), 65: ('tab.bits', 'pro'), 66: ('tab.scan', 'pro'), 67: ('tab.pinf', 'pro'), 68: ('tab.ent', 'pro'), 69: ('split.arrive', 'pro'), 70: ('split.go', 'pro'), 71: ('mma.desc', 'mma'), 72: ('mma.top', 'mma'), 73: ('act.sw', 'epi'), 74: ('act.bar', 'epi'), 75: ('act.st', 'epi'), 76: ('router.unit', 'pro'), 77: ('router.wait', 'pro'), 78: ('router.mma_issued', 'pro'),
    10: ('prod.piece', 'prod'), 11: ('prod.empty_wait', 'prod'), 12: ('prod.h_wait', 'prod'), 13: ('prod.done', 'prod'), 14: ('prod.fc1_done', 'prod'),
    20: ('mma.acc_wait', 'mma'), 21: ('mma.piece', 'mma'), 22: ('mma.full_wait', 'mma'), 23: ('mma.fullp_wait', 'mma'), 24: ('fwd.wait', 'mma'),
    30: ('b.piece', 'b'), 31: ('b.x_ready', 'b'),
    40: ('epi.acc_wait', 'epi'), 41: ('epi.part_store', 'epi'), 43: ('epi.merge_flags', 'epi'), 44: ('epi.merge_loads', 'epi'),
    45: ('epi.act', 'epi'), 46: ('epi.h_publish', 'epi'), 47: ('epi.fc2_tile', 'epi'), 48: ('epi.part_fence', 'epi'), 49: ('epi.done', 'epi'),
    50: ('roles_done', 'tail'), 51: ('t1.rt_wait', 'tail'), 52: ('t1.work', 'tail'), 55: ('allreduce', 'tail'),
    56: ('t1.load_sum', 'tail'), 57: ('t1.send', 'tail'), 58: ('rt.publish', 'tail'), 59: ('t1l.count', 'tail'), 61: ('gate.scores', 'pro'), 62: ('gate.sort', 'pro'), 63: ('gate.topk', 'pro'),
}
WAITS = {'prod.empty_wait', 'prod.h_wait', 'mma.acc_wait', 'mma.full_wait', 'epi.acc_wait', 'epi.merge_flags', 't1.rt_wait', 'pdl_wait'}
ROLES = ['pro', 'prod', 'mma', 'b', 'epi', 'tail']
KINDS = {1: 'routed FC1', 2: 'routed FC2', 3: 'shared FC1', 4: 'shared FC2'}


def dump(buf, grid, meta, path, warps=WARPS, depth=DEPTH):
    """buf: int64 cuda/cpu tensor of the xtrace buffer.  Writes {meta, ctas: [{cta, sm, t0, t1, recs: [[warp, t_ns, site, a0, a1]..]}]}"""
    import torch
    b = buf.detach().cpu()
    hdr = b[:grid * HDR].view(grid, HDR).tolist()
    rec = b[grid * HDR:].view(torch.int32).view(grid, warps, depth, 4)
    clk = rec[..., 0].to(torch.int64) & 0xFFFFFFFF
    site = rec[..., 1].to(torch.int64) & 0xFFFFFFFF
    a0 = rec[..., 2].to(torch.int64) & 0xFFFFFFFF
    a1 = rec[..., 3].to(torch.int64) & 0xFFFFFFFF
    base = min(h[0] for h in hdr if h[0])
    ctas = []
    for c in range(grid):
        g0, c0, g1, c1, sm = hdr[c][:5]
        if not g0 or not g1:
            continue
        k = (g1 - g0) / max(1, (c1 - c0))
        m = site[c] != 0
        w_idx, d_idx = m.nonzero(as_tuple=True)
        rel = (clk[c][m] - (c0 & 0xFFFFFFFF)) % (1 << 32)
        t = (g0 - base) + rel.double() * k
        rows = sorted(zip(w_idx.tolist(), [round(x, 1) for x in t.tolist()], site[c][m].tolist(), a0[c][m].tolist(), a1[c][m].tolist()),
                      key=lambda r: (r[0], r[1]))
        ctas.append(dict(cta=c, sm=sm, t0=g0 - base, t1=g1 - base, clk_ghz=round((c1 - c0) / max(1, g1 - g0), 4), recs=rows))
    Path(path).write_text(json.dumps(dict(meta=meta, ctas=ctas), separators=(',', ':')))


def dec(a0, a1):
    x = (a0 >> 28) & 0xF
    return dict(kind=x, y=a0 & 0xFFFFFFF, z=(a1 >> 16) & 0xFFFF, w=a1 & 0xFFFF)


def build(doc):
    """-> intervals [(cta, warp, role, name, t0, t1, a0, a1, e0, e1)], points [(cta, warp, role, name, t, a0, a1)] (us)"""
    iv, pt = [], []
    for C in doc['ctas']:
        open_ = {}
        for warp, t, site, a0, a1 in C['recs']:
            sid, kind = site & 0xFFF, (site >> 12) & 3
            name, role = SITES.get(sid, (f'site{sid}', 'pro'))
            if role == 'pro' and warp == 0 and sid in (51, 52, 55):
                role = 'tail'
            tu = t / 1e3
            if kind == 1:
                open_[(warp, sid)] = (tu, a0, a1)
            elif kind == 2:
                b = open_.pop((warp, sid), None)
                if b is not None:
                    iv.append((C['cta'], warp, role, name, b[0], tu, b[1], b[2], a0, a1))
            else:
                pt.append((C['cta'], warp, role, name, tu, a0, a1))
    return iv, pt


def q(v, f='{:.2f}'):
    if not v:
        return '-'
    v = sorted(v)
    p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
    return f'{f.format(stt.median(v))} / {f.format(p90)} / {f.format(v[-1])}'


def corr(x, y):
    if len(x) < 3:
        return float('nan')
    mx, my = stt.mean(x), stt.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx > 0 and sy > 0 else float('nan')


def analyze(doc):
    iv, pt = build(doc)
    C = doc['ctas']
    R = {}
    t0s = [c['t0'] / 1e3 for c in C]; t1s = [c['t1'] / 1e3 for c in C]
    res = [b - a for a, b in zip(t0s, t1s)]
    R['residency'] = dict(ctas=len(C), start_spread=max(t0s) - min(t0s), end_first=min(t1s), end_median=stt.median(t1s), end_last=max(t1s),
                          dur_spread=max(res) / max(1e-9, min(res)), sms=len({c['sm'] for c in C}), clk_ghz=stt.median([c['clk_ghz'] for c in C]))
    by = {}
    for r in iv:
        by.setdefault(r[3], []).append(r[5] - r[4])
    R['scopes'] = {k: dict(n=len(v), sum=sum(v), q=q(v)) for k, v in sorted(by.items())}
    # waits per role (sum over CTAs, median per CTA)
    w = {}
    for r in iv:
        if r[3] in WAITS:
            w.setdefault(r[3], {}).setdefault(r[0], 0.0)
            w[r[3]][r[0]] += r[5] - r[4]
    R['waits'] = {k: dict(ctas=len(v), median=stt.median(list(v.values())), max=max(v.values())) for k, v in w.items()}
    # tile-duration correlation within a CTA (consecutive MMA pieces, by kind)
    seqs = {}
    for r in iv:
        if r[3] == 'mma.piece':
            seqs.setdefault(r[0], []).append((r[4], r[5] - r[4]))
    xs, ys = [], []
    for c, s in seqs.items():
        s.sort()
        for (_, a), (_, b) in zip(s, s[1:]):
            xs.append(a); ys.append(b)
    R['tile_corr'] = dict(pairs=len(xs), r=corr(xs, ys))
    # FC1 completion per panel (last h publish), FC2 tiles, tail
    hp = [r for r in iv if r[3] == 'epi.h_publish']
    last = {}
    for r in hp:
        if r[9] & 1:
            last[r[8]] = (r[5], r[0], bool(r[9] & 2))
    R['panels'] = dict(n=len(last), first=min((v[0] for v in last.values()), default=0), median=stt.median([v[0] for v in last.values()]) if last else 0,
                       last=max((v[0] for v in last.values()), default=0),
                       shared=sorted((v[0], k) for k, v in last.items() if v[2]))
    f2 = [r for r in iv if r[3] == 'epi.fc2_tile']
    R['fc2'] = dict(tiles=len(f2), last=max((r[5] for r in f2), default=0),
                    latest=[(round(r[5], 2), r[0], KINDS.get(dec(r[6], r[7])['kind'], '?')) for r in sorted(f2, key=lambda r: r[5])[-5:]])
    rt = [r for r in iv if r[3] == 't1.rt_wait']
    ar = [r for r in iv if r[3] == 'allreduce']
    R['tail'] = dict(t1_wait_end_max=max((r[5] for r in rt), default=0), ar_start_max=max((r[4] for r in ar), default=0),
                     ar_end_max=max((r[5] for r in ar), default=0),
                     roles_done=q([p[4] for p in pt if p[3] == 'roles_done']))
    pro = {}
    for nm in ('pdl_wait', 'gate', 'quant', 'tables'):
        ends = {}
        for r in iv:
            if r[3] == nm:
                ends[r[0]] = max(ends.get(r[0], 0.0), r[5])
        pro[nm + '_end'] = q(list(ends.values()))
    first_piece = {}
    for r in iv:
        if r[3] == 'prod.piece':
            first_piece[r[0]] = min(first_piece.get(r[0], 1e9), r[4])
    pro['first_push'] = q(list(first_piece.values()))
    R['prologue'] = pro
    return R, iv, pt


def show(doc, file=sys.stdout):
    R, iv, pt = analyze(doc)
    m = doc['meta']
    p = lambda *a: print(*a, file=file)
    p(f"xtrace: {m.get('kernel', '?')}  M {m.get('m')}  rank {m.get('rank')}  grid {len(doc['ctas'])}  (times us from the first CTA start)")
    r = R['residency']
    p(f"  residency: {r['ctas']} CTAs on {r['sms']} SMs, start spread {r['start_spread']:.2f}, end first/median/last {r['end_first']:.2f} / "
      f"{r['end_median']:.2f} / {r['end_last']:.2f}, duration spread {r['dur_spread']:.2f}x, SM clock {r['clk_ghz']:.3f} GHz")
    p('  prologue (per-CTA end time, median / p90 / max): ' + '  '.join(f'{k} {v}' for k, v in R['prologue'].items()))
    p('  scopes (duration us, median / p90 / max, n, total):')
    for k, v in R['scopes'].items():
        p(f'    {k:18s} {v["q"]:>26s}   n {v["n"]:6d}   total {v["sum"]:9.1f}')
    p('  waits per CTA (us, median / max over CTAs): ' + '  '.join(f'{k} {v["median"]:.2f}/{v["max"]:.2f}' for k, v in sorted(R['waits'].items())))
    t = R['tile_corr']
    p(f"  tile-duration correlation within a CTA (consecutive MMA pieces): r = {t['r']:.2f} over {t['pairs']} pairs")
    pn = R['panels']
    p(f"  FC1 panels complete (last h publish): n {pn['n']}  first {pn['first']:.2f}  median {pn['median']:.2f}  last {pn['last']:.2f}"
      + (f"  shared panels at {', '.join(f'{a:.2f}' for a, _ in pn['shared'])}" if pn['shared'] else ''))
    f = R['fc2']
    p(f"  FC2 tiles {f['tiles']}, last stored {f['last']:.2f}; latest: " + '  '.join(f'{a} (cta {c}, {k})' for a, c, k in f['latest']))
    tl = R['tail']
    p(f"  tail: T1 row-tile wait ends (max) {tl['t1_wait_end_max']:.2f}, all-reduce start (max) {tl['ar_start_max']:.2f}, end (max) "
      f"{tl['ar_end_max']:.2f}; roles done {tl['roles_done']}")
    return R


def html(docs, out, title='xtrace'):
    """Self-contained interactive panel: per CTA rows (pro / prod / mma / b / epi / tail lanes), scopes as bars, points as ticks."""
    packs = []
    for doc in docs:
        R, iv, pt = analyze(doc)
        names = sorted({r[3] for r in iv} | {p[3] for p in pt})
        nid = {n: i for i, n in enumerate(names)}
        I = [[r[0], ROLES.index(r[2]), nid[r[3]], round(r[4], 3), round(r[5], 3), r[6], r[7], r[8], r[9]] for r in iv]
        Pp = [[p[0], ROLES.index(p[2]), nid[p[3]], round(p[4], 3), p[5], p[6]] for p in pt]
        buf = []
        import io
        s = io.StringIO()
        show(doc, file=s)
        packs.append(dict(meta=doc['meta'], names=names, iv=I, pt=Pp, ctas=[[c['cta'], c['sm'], c['t0'] / 1e3, c['t1'] / 1e3] for c in doc['ctas']],
                          report=s.getvalue()))
    data = json.dumps(packs, separators=(',', ':'))
    page = TEMPLATE.replace('__TITLE__', title).replace('__DATA__', data).replace('__ROLES__', json.dumps(ROLES)).replace('__WAITS__', json.dumps(sorted(WAITS)))
    Path(out).write_text(page)


TEMPLATE = r'''<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* layout: a control strip over one full-width timeline canvas (CTA blocks x warp-role lanes), the analysis report below */
:root{--bg:#f6f7f9;--panel:#ffffff;--fg:#1b1e24;--mut:#636a78;--line:#dfe2e8;--grid:#eceef2;--accent:#2b63d9;--lane:#f0f2f6;
 --ui:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121418;--panel:#181b21;--fg:#e6e8ee;--mut:#98a0ae;--line:#2b303a;--grid:#222731;--accent:#7ea4ff;--lane:#1e222a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121418;--panel:#181b21;--fg:#e6e8ee;--mut:#98a0ae;--line:#2b303a;--grid:#222731;--accent:#7ea4ff;--lane:#1e222a;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.5 var(--ui);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:16px 32px}
header{display:grid;gap:4px;max-width:72rem}h1{font-size:20px;font-weight:600;margin:0;text-wrap:balance}
.sub{color:var(--mut);max-width:68ch}
.bar{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin-block:14px 10px}
.bar label{display:flex;gap:6px;align-items:center;color:var(--mut);font-size:13px}
select,input,button{font:13px var(--ui);color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button:hover{border-color:var(--accent)}:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
#range{font:12px var(--mono);color:var(--mut);font-variant-numeric:tabular-nums}
#wrap{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}canvas{display:block;width:100%;cursor:grab}
#legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin-block:10px}#legend label{display:flex;gap:5px;align-items:center;cursor:pointer;color:var(--mut);font:12px var(--mono)}
#legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
.note{color:var(--mut);font-size:12px;margin-block:4px 16px}
h2{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:18px 0 6px}
.rep{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow-x:auto;min-width:0}
pre{margin:0;padding:12px 14px;font:12px/1.55 var(--mono);font-variant-numeric:tabular-nums;white-space:pre}
#tip{position:fixed;pointer-events:none;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:12px/1.45 var(--mono);z-index:9;max-width:min(440px,90vw);white-space:pre-wrap}
</style>
<header><h1>__TITLE__</h1>
<div class="sub">Intra-kernel phase trace in the manner of Xtrace: every warp role records scope begin/end with the SM cycle counter; each CTA's start and end anchors map cycles onto the GPU-global microsecond timeline. Lanes per CTA: pro (prologue, every warp), prod (TMA producer), mma (tcgen05 issuer), b (B operand), epi (epilogue), tail (T1 + all-reduce).</div></header>
<div class="bar"><label for="sel">Trace <select id="sel"></select></label>
<label for="c0">CTAs <input id="c0" type="number" value="0" min="0" style="width:4.5em"></label><label for="c1">to <input id="c1" type="number" value="24" min="1" style="width:4.5em"></label>
<label for="waitonly"><input id="waitonly" type="checkbox"> Waits only</label><button id="fit" type="button">Fit time</button><span id="range"></span></div>
<div id="wrap"><canvas id="cv" aria-label="timeline"></canvas></div><div id="legend"></div>
<div class="note">Wheel to zoom, drag to pan, hover a bar for its interval and work descriptor. Lighter bars are waits.</div>
<h2>Analysis</h2><div class="rep"><pre id="rep"></pre></div><div id="tip" hidden></div>
<script>
const PACKS=__DATA__, ROLES=__ROLES__, WAITS=new Set(__WAITS__);
const sel=document.getElementById('sel'),cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
PACKS.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`M ${p.meta.m}, rank ${p.meta.rank}`+(p.meta.label?`, ${p.meta.label}`:'');sel.appendChild(o)});
let P=null,view={t0:0,t1:1},hidden=new Set(),hits=[];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dark=()=>css('--bg').toLowerCase()==='#121418';
function color(name){let h=7;for(const ch of name)h=(h*131+ch.charCodeAt(0))>>>0;const hue=(h%330+15)%360;const w=WAITS.has(name);
 return dark()?(w?`hsl(${hue} 45% 38%)`:`hsl(${hue} 60% 58%)`):(w?`hsl(${hue} 70% 80%)`:`hsl(${hue} 58% 46%)`)}
const KIND={1:'routed FC1',2:'routed FC2',3:'shared FC1',4:'shared FC2'};
function desc(a0,a1){const k=(a0>>>28)&15;if(!k)return '';return `${KIND[k]||k}  y ${a0&0xFFFFFFF}  stages ${(a1>>>16)&0xFFFF}..${a1&0xFFFF}`}
function load(i){P=PACKS[i];document.getElementById('rep').textContent=P.report;const tmax=Math.max(...P.ctas.map(c=>c[3]));view={t0:0,t1:tmax*1.01};
 document.getElementById('c1').value=Math.min(P.ctas.length,24);legend();draw()}
function legend(){const L=document.getElementById('legend');L.innerHTML='';P.names.forEach((n,i)=>{const l=document.createElement('label');const cb=document.createElement('input');cb.type='checkbox';cb.id='lg-'+n.replace(/[^a-z0-9]/gi,'_');cb.checked=!hidden.has(n);
 cb.onchange=()=>{cb.checked?hidden.delete(n):hidden.add(n);draw()};l.htmlFor=cb.id;const sw=document.createElement('i');sw.style.background=color(n);l.append(cb,sw,document.createTextNode(n));L.appendChild(l)})}
function draw(){if(!P)return;const c0=Math.max(0,+document.getElementById('c0').value),c1=Math.min(+document.getElementById('c1').value,P.ctas.length);const ctas=P.ctas.slice(c0,c1);
 const lane=11,gap=7,top=24,left=60,W=cv.parentElement.clientWidth,H=top+ctas.length*(ROLES.length*lane+gap)+8,dpr=window.devicePixelRatio||1;
 cv.width=W*dpr;cv.height=H*dpr;cv.style.height=H+'px';ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle=css('--panel');ctx.fillRect(0,0,W,H);
 const X=t=>left+(t-view.t0)/(view.t1-view.t0)*(W-left-8);hits=[];const waitonly=document.getElementById('waitonly').checked;
 const span=view.t1-view.t0,step=Math.pow(10,Math.floor(Math.log10(span/6)));const st=span/step>30?step*5:(span/step>12?step*2:step);
 ctx.font='11px '+css('--mono');ctx.strokeStyle=css('--grid');ctx.fillStyle=css('--mut');
 for(let t=Math.ceil(view.t0/st)*st;t<view.t1;t+=st){const x=X(t);ctx.beginPath();ctx.moveTo(x,top-6);ctx.lineTo(x,H);ctx.stroke();ctx.fillText(t.toFixed(st<1?1:0),x+3,14)}
 ctx.fillText('µs',4,14);
 const row={};ctas.forEach((c,i)=>{row[c[0]]=top+i*(ROLES.length*lane+gap);const y=row[c[0]];ctx.fillStyle=css('--mut');ctx.fillText(`cta ${c[0]}`,4,y+lane*2+1);ctx.fillText(`sm ${c[1]}`,4,y+lane*3+3);
  ctx.fillStyle=css('--lane');const a=Math.max(left,X(c[2])),b=Math.min(W-8,X(c[3]));if(b>a)ctx.fillRect(a,y,b-a,ROLES.length*lane-1)});
 for(const r of P.iv){const y=row[r[0]];if(y===undefined)continue;const nm=P.names[r[2]];if(hidden.has(nm)||(waitonly&&!WAITS.has(nm)))continue;
  if(r[4]<view.t0||r[3]>view.t1)continue;const x0=Math.max(left,X(r[3])),x1=Math.max(Math.min(W-8,X(r[4])),x0+1),yy=y+r[1]*lane;ctx.fillStyle=color(nm);ctx.fillRect(x0,yy+1,x1-x0,lane-2);hits.push([x0,yy,x1,yy+lane,r,0])}
 for(const p of P.pt){const y=row[p[0]];if(y===undefined)continue;const nm=P.names[p[2]];if(hidden.has(nm)||waitonly)continue;if(p[3]<view.t0||p[3]>view.t1)continue;const x=X(p[3]),yy=y+p[1]*lane;ctx.fillStyle=css('--fg');ctx.fillRect(x-1,yy,2,lane);hits.push([x-3,yy,x+3,yy+lane,p,1])}
 document.getElementById('range').textContent=`${view.t0.toFixed(2)} – ${view.t1.toFixed(2)} µs`}
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect(),left=60,W=r.width;const f=Math.min(1,Math.max(0,(e.clientX-r.left-left)/(W-left-8)));const t=view.t0+f*(view.t1-view.t0);
 const k=e.deltaY>0?1.25:0.8;view={t0:t-(t-view.t0)*k,t1:t+(view.t1-t)*k};draw()},{passive:false});
let drag=null;cv.addEventListener('mousedown',e=>{drag={x:e.clientX,v:{...view}}});window.addEventListener('mouseup',()=>drag=null);
window.addEventListener('mousemove',e=>{if(drag){const r=cv.getBoundingClientRect();const dt=(e.clientX-drag.x)/(r.width-68)*(drag.v.t1-drag.v.t0);view={t0:drag.v.t0-dt,t1:drag.v.t1-dt};draw();return}
 const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;const h=hits.find(h=>x>=h[0]&&x<=h[2]&&y>=h[1]&&y<=h[3]);
 if(!h){tip.hidden=true;return}const d=h[4];let s;if(h[5]){s=`${P.names[d[2]]}  cta ${d[0]}  at ${d[3].toFixed(3)} µs`}else{s=`${P.names[d[2]]}  cta ${d[0]} (${ROLES[d[1]]})\n${d[3].toFixed(3)} → ${d[4].toFixed(3)} µs  (${(d[4]-d[3]).toFixed(3)} µs)`+(desc(d[5],d[6])?`\n${desc(d[5],d[6])}`:'')+(d[7]||d[8]?`\nend args ${d[7]} ${d[8]}`:'')}
 tip.textContent=s;tip.hidden=false;tip.style.left=Math.min(e.clientX+12,window.innerWidth-450)+'px';tip.style.top=(e.clientY+12)+'px'});
cv.addEventListener('mouseleave',()=>tip.hidden=true);
['c0','c1','waitonly'].forEach(id=>document.getElementById(id).addEventListener('change',draw));sel.onchange=()=>load(+sel.value);
document.getElementById('fit').onclick=()=>{const tmax=Math.max(...P.ctas.map(c=>c[3]));view={t0:0,t1:tmax*1.01};draw()};
window.addEventListener('resize',draw);window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{legend();draw()});load(0);
</script>'''


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ('show', 'html'):
        print(__doc__); return
    if a[0] == 'show':
        for p in a[1:]:
            show(json.loads(Path(p).read_text()))
    else:
        out = a[a.index('-o') + 1] if '-o' in a else 'xtrace.html'
        paths = [p for i, p in enumerate(a[1:], 1) if p != '-o' and a[i - 1] != '-o']
        html([json.loads(Path(p).read_text()) for p in paths], out, title='tpmoe B200 Trace')
        print('wrote', out)


if __name__ == '__main__':
    main()
