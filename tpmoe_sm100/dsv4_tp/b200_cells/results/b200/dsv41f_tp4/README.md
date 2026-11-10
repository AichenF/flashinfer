# DSV4.1-Flash TP4 on 8x B200 (GPUs 0-3) -- agent B200

suite_b200 commit at writing: `ddfa6e7` (git dir /home/scratch.aichenf_wwfo/suite_b200; nothing pushed).  Common protocol, harness and environment notes: `../COMMON.md`.

**Kernel: v41tp4** (new for this cell) = `port_tp4.py` of tp8v41.cu: NDEV 4, I/4 = 576 rows per rank (9 full FC1 tiles),
FC2 K = 576 as F2T tail-first (main K 512 + a K64 tail as 2 K32 MMAs; bitwise == the ragged path), host `v41tp4/v41tp4host.py`
(main [E][5120][256] + tail [E][128][1280] weight maps).  Source used: `v41tp4/v41tp4t.cu` (patch_tail: LLS on).

## Headline: suite harness, plain-decode real trace (routing_caps/dsv41_real, layers 10/20/30 x steps 32/96, random x; correctness on real x)

Protocol b (L2 flush + gate all-reduce + sleep before every replay; per-rank median, max over ranks), us, mean over the 6 routing cases and the reps; best / ours > 1 = ours faster.

### v4trH (final (c54caf3): v41tp4 (F2T) + LLS, TAILB zeroing guarded, LLS grid guard; clean reps 3f / 4f / 5f (1f: ours slow at M >= 64 on the fresh node; 2f: every arm slow at M >= 64), trt from v4trE)

```
dsv41f_tp4 v4trH: 5 runs (v4trH_1t, v4trH_2t, v4trH_3f, v4trH_4f, v4trH_5f), hosts ['umbriel-b200-038', 'umbriel-b200-040', 'umbriel-b200-092'], ours = ['tp8v41@suite_b200:c54caf3']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 1.40 | 42 | 61.2 [3] | 77.4 [3] | 98.6 [3] | 111.8 [3] | 74.9 | trt 74.9 | 1.224 |
| 16 | 4 | 1.35 | 72 | 84.3 [3] | 98.8 [3] | 122.0 [3] | 133.7 [3] | 102.7 | gc 98.8 | 1.172 |
| 32 | 8 | 1.30 | 115 | 118.6 [3] | 127.6 [3] | 148.2 [3] | 173.1 [3] | 141.3 | gc 127.6 | 1.076 |
| 64 | 16 | 1.33 | 172 | 164.8 [3] | 169.7 [3] | 193.1 [3] | 218.4 [3] | 195.4 | gc 169.7 | 1.030 |
| 128 | 32 | 1.25 | 239 | 222.3 [3] | 218.2 [3] | 245.0 [3] | 288.3 [3] | 248.8 | gc 218.2 | 0.982 |
| 256 | 64 | 1.24 | 294 | 278.5 [3] | 263.9 [3] | 286.1 [3] | 330.9 [3] | 309.4 | gc 263.9 | 0.948 |
| 384 | 96 | 1.25 | 320 | 300.4 [3] | 306.1 [3] | 319.2 [3] | 356.1 [3] | 345.0 | gc 306.1 | 1.019 |
| 512 | 128 | 1.26 | 328 | 320.3 [3] | 350.8 [3] | 347.1 [3] | 361.1 [3] | 373.4 | Mega 347.1 | 1.084 |

correctness (check case, real x):
- trt: rel_q 0.0261..0.0269, cos_q 0.99964..0.99966, rel_exact -, cos_dense 0.99890..0.99893, nonfinite 0, det all, ranks_same all
```

### v4trH (c54caf3, all reps)

```
dsv41f_tp4 v4trH: 7 runs (v4trH_1f, v4trH_1t, v4trH_2f, v4trH_2t, v4trH_3f, v4trH_4f, v4trH_5f), hosts ['umbriel-b200-038', 'umbriel-b200-040', 'umbriel-b200-092'], ours = ['tp8v41@suite_b200:c54caf3']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 1.40 | 42 | 61.5 [5] | 78.1 [5] | 99.3 [5] | 112.5 [5] | 74.9 | trt 74.9 | 1.218 |
| 16 | 4 | 1.35 | 72 | 84.6 [5] | 99.3 [5] | 122.4 [5] | 134.3 [5] | 102.7 | gc 99.3 | 1.174 |
| 32 | 8 | 1.30 | 115 | 118.8 [5] | 127.9 [5] | 148.3 [5] | 173.4 [5] | 141.3 | gc 127.9 | 1.077 |
| 64 | 16 | 1.33 | 172 | 170.7 [5] | 171.4 [5] | 193.1 [5] | 218.8 [5] | 195.4 | gc 171.4 | 1.004 |
| 128 | 32 | 1.25 | 239 | 231.4 [5] | 220.4 [5] | 244.9 [5] | 288.0 [5] | 248.8 | gc 220.4 | 0.953 |
| 256 | 64 | 1.24 | 294 | 292.0 [5] | 264.1 [5] | 285.2 [5] | 328.0 [5] | 309.4 | gc 264.1 | 0.904 |
| 384 | 96 | 1.25 | 320 | 310.7 [5] | 307.1 [5] | 319.1 [5] | 355.9 [5] | 345.0 | gc 307.1 | 0.988 |
| 512 | 128 | 1.26 | 328 | 332.4 [5] | 350.9 [5] | 347.5 [5] | 362.0 [5] | 373.4 | Mega 347.5 | 1.046 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 5.52e-05..0.000204, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0261..0.0269, cos_q 0.99964..0.99966, rel_exact -, cos_dense 0.99890..0.99893, nonfinite 0, det all, ranks_same all
```

### v4trE (v41tp4 + LLS (40eb85a), reps 1f + 3f (node 040; rep 2f had every arm slow at M >= 64, ours +12% at M512))

```
dsv41f_tp4 v4trE: 4 runs (v4trE_1f, v4trE_1t, v4trE_2t, v4trE_3f), hosts ['umbriel-b200-040'], ours = ['tp8v41@suite_b200:6081398', 'tp8v41@suite_b200:88f4b3d']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 1.40 | 42 | 60.9 | 76.6 | 98.5 | 109.7 | 74.9 | trt 74.9 | 1.230 |
| 16 | 4 | 1.35 | 72 | 84.4 | 98.5 | 120.8 | 132.9 | 102.7 | gc 98.5 | 1.167 |
| 32 | 8 | 1.30 | 115 | 117.9 | 126.7 | 147.1 | 171.7 | 141.3 | gc 126.7 | 1.074 |
| 64 | 16 | 1.33 | 172 | 164.5 | 168.8 | 195.1 | 219.4 | 195.4 | gc 168.8 | 1.026 |
| 128 | 32 | 1.25 | 239 | 221.0 | 217.6 | 245.6 | 285.9 | 248.8 | gc 217.6 | 0.984 |
| 256 | 64 | 1.24 | 294 | 278.0 | 264.5 | 285.8 | 329.2 | 309.4 | gc 264.5 | 0.951 |
| 384 | 96 | 1.25 | 320 | 301.5 | 309.6 | 319.7 | 357.6 | 345.0 | gc 309.6 | 1.027 |
| 512 | 128 | 1.26 | 328 | 321.1 | 347.2 | 348.8 | 358.0 | 373.4 | gc 347.2 | 1.081 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 5.52e-05..0.000204, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0261..0.0269, cos_q 0.99964..0.99966, rel_exact -, cos_dense 0.99890..0.99893, nonfinite 0, det all, ranks_same all
```

### v4trE (40eb85a, all 3 reps)

```
dsv41f_tp4 v4trE: 5 runs (v4trE_1f, v4trE_1t, v4trE_2f, v4trE_2t, v4trE_3f), hosts ['umbriel-b200-040'], ours = ['tp8v41@suite_b200:6081398', 'tp8v41@suite_b200:88f4b3d']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 1.40 | 42 | 61.0 [3] | 76.7 [3] | 98.3 [3] | 109.8 [3] | 74.9 | trt 74.9 | 1.229 |
| 16 | 4 | 1.35 | 72 | 84.6 [3] | 98.6 [3] | 120.7 [3] | 133.1 [3] | 102.7 | gc 98.6 | 1.166 |
| 32 | 8 | 1.30 | 115 | 118.1 [3] | 126.6 [3] | 147.2 [3] | 171.8 [3] | 141.3 | gc 126.6 | 1.072 |
| 64 | 16 | 1.33 | 172 | 165.7 [3] | 170.1 [3] | 194.0 [3] | 217.8 [3] | 195.4 | gc 170.1 | 1.026 |
| 128 | 32 | 1.25 | 239 | 224.4 [3] | 222.4 [3] | 246.0 [3] | 283.3 [3] | 248.8 | gc 222.4 | 0.991 |
| 256 | 64 | 1.24 | 294 | 285.4 [3] | 267.8 [3] | 285.7 [3] | 326.3 [3] | 309.4 | gc 267.8 | 0.938 |
| 384 | 96 | 1.25 | 320 | 310.6 [3] | 315.4 [3] | 319.9 [3] | 356.2 [3] | 345.0 | gc 315.4 | 1.016 |
| 512 | 128 | 1.26 | 328 | 333.6 [3] | 350.7 [3] | 348.4 [3] | 359.2 [3] | 373.4 | Mega 348.4 | 1.044 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 5.52e-05..0.000204, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0261..0.0269, cos_q 0.99964..0.99966, rel_exact -, cos_dense 0.99890..0.99893, nonfinite 0, det all, ranks_same all
```

### v4tr (first run (node 026))

```
dsv41f_tp4 v4tr: 4 runs (v4tr_1f, v4tr_1t, v4tr_2f, v4tr_2t), hosts ['umbriel-b200-026'], ours = ['tp8v41@suite_b200:06136eb', 'tp8v41@suite_b200:9761840']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 1.40 | 42 | 61.0 | 78.2 | 99.3 | 111.8 | 75.9 | trt 75.9 | 1.246 |
| 16 | 4 | 1.35 | 72 | 85.0 | 100.4 | 122.7 | 134.4 | 103.3 | gc 100.4 | 1.181 |
| 32 | 8 | 1.30 | 115 | 119.4 | 130.4 | 149.4 | 173.6 | 142.2 | gc 130.4 | 1.092 |
| 64 | 16 | 1.33 | 172 | 170.0 | 178.0 | 196.4 | 221.7 | 197.3 | gc 178.0 | 1.047 |
| 128 | 32 | 1.25 | 239 | 229.6 | 231.8 | 246.5 | 289.3 | 255.5 | gc 231.8 | 1.009 |
| 256 | 64 | 1.24 | 294 | 287.7 | 281.8 | 294.6 | 335.4 | 310.6 | gc 281.8 | 0.980 |
| 384 | 96 | 1.25 | 320 | 313.0 | 316.6 | 330.4 | 363.7 | 350.7 | gc 316.6 | 1.011 |
| 512 | 128 | 1.26 | 328 | 339.6 | 367.0 | 358.3 | 371.5 | 376.1 | Mega 358.3 | 1.055 |

correctness (check case, real x):
- ours: rel_q 0.0327..0.0347, cos_q 0.99941..0.99948, rel_exact 5.52e-05..0.000204, cos_dense 0.99869..0.99872, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- Mega: rel_q 0.0264..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- fiMega: rel_q 0.0262..0.0266, cos_q 0.99965..0.99966, rel_exact -, cos_dense 0.99891..0.99895, nonfinite 0, det all
- trt: rel_q 0.0261..0.0269, cos_q 0.99964..0.99966, rel_exact -, cos_dense 0.99890..0.99893, nonfinite 0, det all, ranks_same all
```

## Correctness gate of the final build (back-to-back included)

my harness `sb.py --b2b 12 --b2b-reps 4 --check 1` on the final source (log `logs/gate_v4trH.log`): per M the check case (rel vs the numerics emulation, ranks bitwise identical, deterministic) and 12-call back-to-back bursts on distinct inputs (eager with rank-skewed starts / one rank drifting / one CUDA graph of the 12 calls), every output bitwise vs its sync-separated reference on every rank (mismatching elements; 0 = clean):

```
RES M= 8 ours: rel 0.00000 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 16 ours: rel 0.00002 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 32 ours: rel 0.00016 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 64 ours: rel 0.00001 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 128 ours: rel 0.00004 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 256 ours: rel 0.00001 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 384 ours: rel 0.00010 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 512 ours: rel 0.00007 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
```

## Secondary: my harness (sb.py), same trace

### c41t4tr (my harness, v41tp4 (before LLS))

```
c41t4tr routing=trace protocol=b: us, mean of 2 reps (spread max-min)
    M           ours            trt             gc           mega best-base best/ours  recv    U
    8     60.4 ( 0.0)    75.1 ( 0.6)    77.5 ( 1.4)    97.7 ( 0.7)       trt     1.243  1.40   42
   16     84.1 ( 1.0)   103.7 ( 1.7)    98.3 ( 0.0)   119.8 ( 0.0)        gc     1.169  1.35   72
   32    118.4 ( 0.7)   138.8 ( 0.3)   126.2 ( 0.3)   147.3 ( 0.2)        gc     1.065  1.30  115
   64    169.4 ( 1.7)   194.4 ( 0.4)   175.5 ( 4.6)   193.0 ( 3.8)        gc     1.036  1.33  172
  128    236.6 ( 2.9)   255.2 ( 0.3)   227.5 ( 0.4)   244.5 ( 2.2)        gc     0.961  1.25  239
  256    299.9 ( 2.9)   315.4 ( 4.8)   277.9 ( 8.9)   287.5 ( 5.3)        gc     0.926  1.24  294
  384    333.2 ( 8.4)   353.7 ( 4.0)   317.4 (11.1)   322.5 ( 1.3)        gc     0.952  1.25  320
  512    359.4 ( 8.1)   384.0 ( 6.8)   350.1 (20.1)   354.0 (12.3)        gc     0.974  1.26  328
mean best-baseline/ours 1.041 over 8 M; ours wins at 4
```

## Optimizations

Kept (bitwise, 2 agreeing reps): F2T K64 tails (-3..-4% at M >= 64, commit db4b19b), LLS (commit 4c9634b): M64 / 128 / 256 / 512
-2.0..-2.1 / -2.8..-3.1 / -4.4..-5.6 / -10.2 us (M384 -8.3).
Guards: TAILB counter zeroing only in TAILB builds, lls_allreduce traps if gridDim != TPB2_GRID (c54caf3).  This lineage has no SFBI item
buffers (no analogue of the v77 SFBI hazard).

TP4 on the new source (port_tp4h2.py -> v41tp4h2/, SB_V41TP4_DIR=v41tp4h2; F2E / F2D / F2SB / F2P forbidden at TP4): correct
(b2b 12 gate at every M: 0 mismatches) but cross-process vs this base M64 / 128 / 256 / 384 / 512 +0.7..+1.5 / +2.7 / ~0 / -1.1..-1.7 /
-2.5..-2.7 us -> TP4 stays on v41tp4.  H2 forced at TP4 (F2H at NDEV 4): correct (rel 1e-4 vs ref_rank_h, b2b 0) but M384 / 512 +9.3 / +6.5 us
(the h all-gather of 4 ranks outweighs the K-576 FC2 it replaces) -> off.
Refuted: oneshot up to M64 at TP4 (slower than RS+AG from M64), T1PW 8 (+9 us at M512) / T1PW 2 / T1STATIC 0 (mixed;
one T1STATIC-0 process hit the slow-timing state).  Baseline note: my harness's gc runs this cell faster than the suite's gc at
large M (M512 323.6 us, k101_* logs, vs 345-358 in the suite), so the M384 / M512 wins here depend on the baseline harness.

## Timing caveat

In some processes ours (v77 and tp8v41 alike) runs 5-15% slow at the larger M -- for one M or for the rest of the process -- while gc in the same process is normal (suite f8tr_1f / f8trL_1f / f8trX_1f, my c4ftrB_1f / k064 / k091 / k097_2).  It is not reproducible on demand: 4 repeated check-1 processes, fresh-build processes and pre-capture eager calls were all clean, and the post-timing graph output stays bitwise (post_mism 0), so it is a timing state, not a wrong-result mode.  The trigger is not understood.  The per-rep values are in each result.json (`*_1f` vs `*_2f` / `*_3f`); prefer the clean reps when comparing builds.  Node-to-node spread is ~4%; B200 event timestamps are quantized to ~2.05 us.

Reproduce (computelab, a B200 x8 hold set up with `JOB=<hold> tpmoe_suite/env/node_setup.sh sm100`):
```
cd /home/scratch.aichenf_wwfo/suite_b200
# suite harness, 2 reps (rep 1: check 1, arms fwd; rep 2: check 0, arms reversed) + trt in its own process
JOB=<hold> ./suite.sh dsv41 4 <tag> real-trace:/home/scratch.aichenf_wwfo/routing_caps/dsv41_real
python3 -I stable.py results/b200/dsv41f_tp4 <tag>
# my harness (paired A/B of a build change: --flags2=..., bitwise check ours2_vs_ours)
JOB=<hold> GPUS=0,1,2,3 ./sctr.sh <out> python sb.py --model dsv41 --tp 4 --ms 8 16 32 64 128 256 384 512 --arms ours gc mega --routing trace --check 1 --proto b
```
Builds are JIT (torch cpp_extension, sm_100a) into the node-local build dir on first use (~1-4 min per M build).
