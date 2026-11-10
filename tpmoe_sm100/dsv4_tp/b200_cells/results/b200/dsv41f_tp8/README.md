# DSV4.1-Flash TP8 on 8x B200 -- agent B200

suite_b200 commit at writing: `ddfa6e7` (git dir /home/scratch.aichenf_wwfo/suite_b200; nothing pushed).  Common protocol, harness and environment notes: `../COMMON.md`.

**Kernel (since 11-10): tp8v41h2** = tp8_v41_b300 @3668057 (tp8v41h2/tp8v41.cu + v41host.py, copied read-only)
+ the suite_b200 LLS grid trap (tp8v41g.cu, mk_g.py): LLS on, NMAX 32 and GATHER 0 for M >= 256, H2 (FC2 sharded on H + epoch-tagged
h exchange, fixed HGN=32 layout, T1PAR tail) for M >= 128 by default (TP8_F2H_MINM), F2H + ACTX forbidden.  Earlier steps: 2625d29 (b690545),
10ddd15 (7d8e42a).  Gate on 3668057: --b2b 12 every M + --b2b-mix 0 mismatches; stress M64..512 120k calls per rank 0 bad (st41e).  H2 numerics (full-K fp32 FC2, one bf16 rounding
per slot) are checked against ref_rank_h; arm_tp8v41 sets up the all-gathered H-sharded W2 and the symmetric gathered-h buffer.
Gate on this build: --b2b 12 (skew / drift / graph) at every M 8..512 and --b2b-mix: 0 mismatches; high-count stress (1500 x 8 eager +
graph, 2 cases): default builds M64..512 120k, H2 forced at M64 / 128 48k, the non-H2 base at M256..512 72k calls per rank, 0 bad
(results/b200/_stress).  B200 TP8 same-run A/B H2 vs LLS-only (both this source, 2 swapped reps): M128 -1.2 / -0.7, M256 -5.8 / -5.1,
M384 -8.5 / -8.9, M512 -14.3 / -13.6 us, M64 +7.0.  Note: at M128 (non-H2 path) this base reads ~2-3 us slower than the previous one.
Previous base (SB_TP8V41_DIR=tp8v41 SB_V41SRC=tp8v41t.cu): **tp8v41** = the B300-TP8 agent's DSV4.1 TP8 kernel (tp8_v41_b300 @ee6b585: port_v41.py of k3_b300/L/k3h.cu; copied
read-only into `tp8v41/`) built for sm_100a, routed-only, in-kernel one-shot (M <= 32) / RS+AG all-reduce.  Source used:
`tp8v41/tp8v41t.cu` = `patch_tail.py` (generic tail knobs: LLS on, TAILOV / TAILB off) of `tp8v41/tp8v41.cu`.  Host `arm_tp8v41.py`
(per-M exact builds `v41host.spec_flags(M, M, 8)` + ONESHOT_MAXM 32, LLS slots armed to the sentinel once).

## Headline: suite harness, plain-decode real trace (routing_caps/dsv41_real, layers 10/20/30 x steps 32/96, random x; correctness on real x)

Protocol b (L2 flush + gate all-reduce + sleep before every replay; per-rank median, max over ranks), us, mean over the 6 routing cases and the reps; best / ours > 1 = ours faster.

### v8trP (final: tp8v41h2 = tp8_v41_b300 @3668057 (LLS, NMAX 32 + GATHER 0 at M >= 256, H2 + T1PAR from M >= 128) + LLS grid trap, 3 reps, trt from v8trD (run 00:42 11-10 on the then-uncommitted 3668057 sources, committed next))

```
dsv41f_tp8 v8trP: 5 runs (v8trP_1f, v8trP_1t, v8trP_2f, v8trP_2t, v8trP_3f), hosts ['umbriel-b200-069', 'umbriel-b200-074'], ours = ['tp8v41h2@suite_b200:7d8e42a(F2H_MINM=host default: 256 at 2625d29)']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.61 | 42 | 52.9 [3] | 66.1 [3] | 84.0 [3] | 141.4 [3] | 65.4 | trt 65.4 | 1.237 |
| 16 | 2 | 1.74 | 72 | 67.1 [3] | 78.7 [3] | 100.3 [3] | 111.5 [3] | 83.9 | gc 78.7 | 1.172 |
| 32 | 4 | 1.50 | 115 | 86.2 [3] | 94.1 [3] | 117.2 [3] | 128.0 [3] | 111.6 | gc 94.1 | 1.091 |
| 64 | 8 | 1.63 | 172 | 109.7 [3] | 113.4 [3] | 137.4 [3] | 153.0 [3] | 156.5 | gc 113.4 | 1.034 |
| 128 | 16 | 1.46 | 239 | 135.3 [3] | 137.0 [3] | 158.8 [3] | 184.1 [3] | 198.3 | gc 137.0 | 1.013 |
| 256 | 32 | 1.47 | 294 | 162.7 [3] | 160.7 [3] | 178.0 [3] | 195.9 [3] | 254.8 | gc 160.7 | 0.988 |
| 384 | 48 | 1.43 | 320 | 181.1 [3] | 183.1 [3] | 192.8 [3] | 209.2 [3] | 267.4 | gc 183.1 | 1.011 |
| 512 | 64 | 1.45 | 328 | 192.5 [3] | 199.8 [3] | 204.4 [3] | 210.0 [3] | 284.6 | gc 199.8 | 1.038 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99942..0.99948, rel_exact 4.79e-05..0.00365, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0263..0.0271, cos_q 0.99963..0.99966, rel_exact -, cos_dense 0.99889..0.99893, nonfinite 0, det all, ranks_same all
```

### v8trN (tp8v41h2 at 2625d29 (H2 for M >= 256), 3 reps, trt from v8trD)

```
dsv41f_tp8 v8trN: 5 runs (v8trN_1f, v8trN_1t, v8trN_2f, v8trN_2t, v8trN_3f), hosts ['umbriel-b200-069', 'umbriel-b200-074'], ours = ['tp8v41h2@suite_b200:1020ce9(F2H_MINM=off)']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.61 | 42 | 53.1 [3] | 66.1 [3] | 83.6 [3] | 142.2 [3] | 65.4 | trt 65.4 | 1.232 |
| 16 | 2 | 1.74 | 72 | 67.0 [3] | 78.7 [3] | 100.5 [3] | 110.9 [3] | 83.9 | gc 78.7 | 1.175 |
| 32 | 4 | 1.50 | 115 | 86.0 [3] | 93.9 [3] | 117.2 [3] | 127.7 [3] | 111.6 | gc 93.9 | 1.093 |
| 64 | 8 | 1.63 | 172 | 109.6 [3] | 113.2 [3] | 137.3 [3] | 152.9 [3] | 156.5 | gc 113.2 | 1.033 |
| 128 | 16 | 1.46 | 239 | 142.3 [3] | 136.8 [3] | 158.8 [3] | 183.6 [3] | 198.3 | gc 136.8 | 0.961 |
| 256 | 32 | 1.47 | 294 | 170.7 [3] | 158.4 [3] | 177.6 [3] | 196.2 [3] | 254.8 | gc 158.4 | 0.928 |
| 384 | 48 | 1.43 | 320 | 186.1 [3] | 181.0 [3] | 193.4 [3] | 210.3 [3] | 267.4 | gc 181.0 | 0.973 |
| 512 | 64 | 1.45 | 328 | 195.9 [3] | 199.1 [3] | 204.9 [3] | 208.2 [3] | 284.6 | gc 199.1 | 1.017 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 4.79e-05..0.00365, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0263..0.0271, cos_q 0.99963..0.99966, rel_exact -, cos_dense 0.99889..0.99893, nonfinite 0, det all, ranks_same all
```

### v8trG (previous base (c54caf3): tp8v41 @ee6b585 + patch_tail (LLS, TAILB zeroing guarded, LLS grid guard), 3 reps, trt from v8trD)

```
dsv41f_tp8 v8trG: 5 runs (v8trG_1f, v8trG_1t, v8trG_2f, v8trG_2t, v8trG_3f), hosts ['umbriel-b200-069', 'umbriel-b200-092'], ours = ['tp8v41@suite_b200:c54caf3']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.61 | 42 | 51.2 [3] | 66.4 [3] | 83.8 [3] | 141.5 [3] | 65.4 | trt 65.4 | 1.278 |
| 16 | 2 | 1.74 | 72 | 65.5 [3] | 78.6 [3] | 100.2 [3] | 111.7 [3] | 83.9 | gc 78.6 | 1.199 |
| 32 | 4 | 1.50 | 115 | 84.5 [3] | 93.9 [3] | 116.7 [3] | 127.2 [3] | 111.6 | gc 93.9 | 1.111 |
| 64 | 8 | 1.63 | 172 | 107.2 [3] | 112.9 [3] | 136.4 [3] | 151.7 [3] | 156.5 | gc 112.9 | 1.053 |
| 128 | 16 | 1.46 | 239 | 139.2 [3] | 136.9 [3] | 160.0 [3] | 182.7 [3] | 198.3 | gc 136.9 | 0.984 |
| 256 | 32 | 1.47 | 294 | 174.4 [3] | 159.2 [3] | 178.9 [3] | 196.3 [3] | 254.8 | gc 159.2 | 0.913 |
| 384 | 48 | 1.43 | 320 | 194.3 [3] | 180.8 [3] | 194.7 [3] | 210.7 [3] | 267.4 | gc 180.8 | 0.931 |
| 512 | 64 | 1.45 | 328 | 212.4 [3] | 198.8 [3] | 207.2 [3] | 210.9 [3] | 284.6 | gc 198.8 | 0.936 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 4.79e-05..0.000201, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0263..0.0271, cos_q 0.99963..0.99966, rel_exact -, cos_dense 0.99889..0.99893, nonfinite 0, det all, ranks_same all
```

### v8trD (tp8v41 + LLS (40eb85a), 3 reps (node 069))

```
dsv41f_tp8 v8trD: 5 runs (v8trD_1f, v8trD_1t, v8trD_2f, v8trD_2t, v8trD_3f), hosts ['umbriel-b200-069'], ours = ['tp8v41@suite_b200:6081398']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.61 | 42 | 52.5 [3] | 67.9 [3] | 85.0 [3] | 138.0 [3] | 65.4 | trt 65.4 | 1.248 |
| 16 | 2 | 1.74 | 72 | 66.6 [3] | 81.3 [3] | 102.8 [3] | 113.3 [3] | 83.9 | gc 81.3 | 1.220 |
| 32 | 4 | 1.50 | 115 | 85.5 [3] | 96.7 [3] | 118.4 [3] | 128.7 [3] | 111.6 | gc 96.7 | 1.132 |
| 64 | 8 | 1.63 | 172 | 108.7 [3] | 115.8 [3] | 138.0 [3] | 153.7 [3] | 156.5 | gc 115.8 | 1.065 |
| 128 | 16 | 1.46 | 239 | 140.3 [3] | 138.6 [3] | 160.3 [3] | 184.7 [3] | 198.3 | gc 138.6 | 0.988 |
| 256 | 32 | 1.47 | 294 | 175.1 [3] | 160.1 [3] | 177.4 [3] | 196.1 [3] | 254.8 | gc 160.1 | 0.914 |
| 384 | 48 | 1.43 | 320 | 194.7 [3] | 177.5 [3] | 190.9 [3] | 208.4 [3] | 267.4 | gc 177.5 | 0.912 |
| 512 | 64 | 1.45 | 328 | 212.6 [3] | 196.8 [3] | 204.1 [3] | 207.8 [3] | 284.6 | gc 196.8 | 0.926 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 4.79e-05..0.000201, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0263..0.0271, cos_q 0.99963..0.99966, rel_exact -, cos_dense 0.99889..0.99893, nonfinite 0, det all, ranks_same all
```

### v8tr (tp8v41 + LLS, first run (node 069))

```
dsv41f_tp8 v8tr: 4 runs (v8tr_1f, v8tr_1t, v8tr_2f, v8tr_2t), hosts ['umbriel-b200-069'], ours = ['tp8v41@suite_b200:06136eb', 'tp8v41@suite_b200:9761840']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.61 | 42 | 52.4 | 67.8 | 85.0 | 138.1 | 65.4 | trt 65.4 | 1.249 |
| 16 | 2 | 1.74 | 72 | 67.1 | 81.7 | 102.6 | 113.3 | 84.1 | gc 81.7 | 1.219 |
| 32 | 4 | 1.50 | 115 | 85.5 | 97.0 | 118.6 | 128.8 | 111.6 | gc 97.0 | 1.134 |
| 64 | 8 | 1.63 | 172 | 109.2 | 115.7 | 137.9 | 154.1 | 156.5 | gc 115.7 | 1.060 |
| 128 | 16 | 1.46 | 239 | 140.3 | 138.5 | 160.2 | 184.7 | 197.5 | gc 138.5 | 0.987 |
| 256 | 32 | 1.47 | 294 | 175.0 | 160.8 | 177.1 | 196.8 | 254.1 | gc 160.8 | 0.919 |
| 384 | 48 | 1.43 | 320 | 195.0 | 178.0 | 190.6 | 209.3 | 265.4 | gc 178.0 | 0.913 |
| 512 | 64 | 1.45 | 328 | 213.0 | 197.4 | 203.6 | 208.4 | 282.7 | gc 197.4 | 0.927 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 4.79e-05..0.000201, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0263..0.0271, cos_q 0.99963..0.99966, rel_exact -, cos_dense 0.99889..0.99893, nonfinite 0, det all, ranks_same all
```

## Correctness gate of the final build (back-to-back included)

my harness `sb.py --b2b 12 --b2b-reps 4 --check 1` on the final source (log `logs/gate_v8trG.log`): per M the check case (rel vs the numerics emulation, ranks bitwise identical, deterministic) and 12-call back-to-back bursts on distinct inputs (eager with rank-skewed starts / one rank drifting / one CUDA graph of the 12 calls), every output bitwise vs its sync-separated reference on every rank (mismatching elements; 0 = clean):

```
RES M= 8 ours: rel 0.00001 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 16 ours: rel 0.00002 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 32 ours: rel 0.00001 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 64 ours: rel 0.00002 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 128 ours: rel 0.00002 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 256 ours: rel 0.00003 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 384 ours: rel 0.00003 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 512 ours: rel 0.00005 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
```

## Secondary: my harness (sb.py), same trace

### c41t8tr (my harness, tp8v41 (before LLS))

```
c41t8tr routing=trace protocol=b: us, mean of 2 reps (spread max-min)
    M           ours            trt             gc           mega best-base best/ours  recv    U
    8     50.3 ( 1.0)    63.0 ( 0.3)    66.4 ( 1.6)    83.8 ( 0.4)       trt     1.253  1.61   42
   16     65.3 ( 0.3)    81.7 ( 0.5)    77.7 ( 1.1)    99.0 ( 0.6)        gc     1.188  1.74   72
   32     85.0 ( 0.0)   112.7 ( 2.1)    93.5 ( 0.9)   117.2 ( 0.3)        gc     1.099  1.50  115
   64    107.3 ( 0.3)   155.6 ( 2.9)   112.4 ( 0.3)   136.4 ( 1.0)        gc     1.048  1.63  172
  128    139.7 ( 0.8)   197.9 ( 2.1)   136.6 ( 0.7)   158.8 ( 0.2)        gc     0.977  1.46  239
  256    177.6 ( 0.2)   255.1 ( 1.0)   158.6 ( 1.7)   175.6 ( 0.4)        gc     0.893  1.47  294
  384    198.8 ( 1.7)   263.4 ( 0.3)   180.0 ( 2.4)   189.4 ( 0.7)        gc     0.905  1.43  320
  512    220.6 ( 0.2)   284.9 ( 3.1)   198.8 ( 1.0)   203.3 ( 2.3)        gc     0.901  1.45  328
mean best-baseline/ours 1.033 over 8 M; ours wins at 4
```

## Optimizations

Kept (bitwise, 2 agreeing reps): LLS sentinel all-reduce messages (commit 4c9634b; also usable by the B300 builds -- patch_tail.py
is shape-generic): M64 / 128 / 256 / 512 -1.6..-2.2 / -2.8..-2.9 / -3.9..-5.1 / -8.5..-8.9 us.
Guards: TAILB counter zeroing only in TAILB builds, lls_allreduce traps if gridDim != TPB2_GRID (c54caf3).  This lineage has no SFBI item
buffers (no analogue of the v77 SFBI hazard).

Refuted (tail): TAILOV (T1 || owner-reduce overlap), TAILB (sys-scope fences cost 10-15 us with NVLink writes in flight),
MC=2, GATHER=1 at M >= 384; B200 retune of the T1 knobs: T1PW 8 (+7 us at M512), T1PW 2 / T1STATIC 0 (within +-2 us,
mixed).  The FC2 per-stage / epilogue, hot-expert panel sharing and 2-CTA items belong to the B300-TP8 agent.
Non-H2 B200 regression (M32..128 +1.7..3.9 us vs ee6b585, bisected in 2651b65 to the F2H commits 420746a / 61ddeb0): tp8_v41_b300 8bbb174
(all F2H params in a struct at the end of Params, pre-F2H layout for non-F2H builds) re-measured 11-10 (tp8v41h2/mk_ab.sh -> tp8v41g8.cu):
bitwise == 3668057 at every M, timing neutral (M8..512 within -1.1..+0.5 us, 2 swapped reps), H2-off vs ee6b585 still +1.7..2.1 / +3.2..3.5 /
+2.4..2.8 us at M32 / 64 / 128 -> not the cause; final row stays 37b469a (no ee6b585 dispatch for M32 / 64, coordinator decision).

## Timing caveat

In some processes ours (v77 and tp8v41 alike) runs 5-15% slow at the larger M -- for one M or for the rest of the process -- while gc in the same process is normal (suite f8tr_1f / f8trL_1f / f8trX_1f, my c4ftrB_1f / k064 / k091 / k097_2).  It is not reproducible on demand: 4 repeated check-1 processes, fresh-build processes and pre-capture eager calls were all clean, and the post-timing graph output stays bitwise (post_mism 0), so it is a timing state, not a wrong-result mode.  The trigger is not understood.  The per-rep values are in each result.json (`*_1f` vs `*_2f` / `*_3f`); prefer the clean reps when comparing builds.  Node-to-node spread is ~4%; B200 event timestamps are quantized to ~2.05 us.

Reproduce (computelab, a B200 x8 hold set up with `JOB=<hold> tpmoe_suite/env/node_setup.sh sm100`):
```
cd /home/scratch.aichenf_wwfo/suite_b200
# suite harness, 2 reps (rep 1: check 1, arms fwd; rep 2: check 0, arms reversed) + trt in its own process
JOB=<hold> ./suite.sh dsv41 8 <tag> real-trace:/home/scratch.aichenf_wwfo/routing_caps/dsv41_real
python3 -I stable.py results/b200/dsv41f_tp8 <tag>
# my harness (paired A/B of a build change: --flags2=..., bitwise check ours2_vs_ours)
JOB=<hold> GPUS=0,1,2,3,4,5,6,7 ./sctr.sh <out> python sb.py --model dsv41 --tp 8 --ms 8 16 32 64 128 256 384 512 --arms ours gc mega --routing trace --check 1 --proto b
```
Builds are JIT (torch cpp_extension, sm_100a) into the node-local build dir on first use (~1-4 min per M build).
