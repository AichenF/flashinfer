# DSV4-Flash TP8 on 8x B200 -- agent B200

suite_b200 commit at writing: `ddfa6e7` (git dir /home/scratch.aichenf_wwfo/suite_b200; nothing pushed).  Common protocol, harness and environment notes: `../COMMON.md`.

**Kernel: v77 at NDEV 8** (new for this cell): the v77 source with `patch_ndev.py` (TPB2_NDEV=8: 8 ranks, I/8 = 256 rows
per rank, row tiles owned 2 per rank in the RS+AG all-reduce) + the same routed-only / IDV / LLS / XRLX patches as the TP4 cell.
Host `arm_v77.py` with `-DTPB2_NDEV=8`.

**Since 11-10 (6a01f29): M >= 384 runs v4ft** (`arm_disp.py`; suite adapter default, `SB_DSV4F_DISP=0` = v77 at every M): v4ft =
suite_b300b/s3/port_v4f.py over tp8_v41_b300 e066ec9 (`v4ft/v4fti.cu`, md5-identical regen; LLS + grid trap, NMAX 32 and GATHER 0 from
M 256, the B300b cell flags ACTX + T1PW2 at M384 / 512, F2H off), host `v4ft/s3host.py`, arm `arm_v4ft.py`; v77 and v4ft keep separate
workspaces on one stream.  Numerics at M >= 384 = the tp8v41 lineage's (fp32 routing weights, checked against s3host.ref_rank).

## Headline: suite harness, plain-decode real trace (routing_caps/dsv4f_real, layers 10/20/30 x steps 32/96, random x; correctness on real x)

Protocol b (L2 flush + gate all-reduce + sleep before every replay; per-rank median, max over ranks), us, mean over the 6 routing cases and the reps; best / ours > 1 = ours faster.

### f8trV (final (6a01f29): v77-ndev8 for M < 384, v4ft (e066ec9 port, ACTX + T1PW2) for M >= 384, 3 reps 1f/2f/3f, trt from f8trX)

```
dsv4f_tp8 f8trV: 5 runs (f8trV_1f, f8trV_1t, f8trV_2f, f8trV_2t, f8trV_3f), hosts ['umbriel-b200-069', 'umbriel-b200-074'], ours = ['v77-ndev8<384<=v4ft(e066ec9)@suite_b200:6a01f29']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.67 | 42 | 44.6 [3] | 57.4 [3] | 69.7 [3] | 124.9 [3] | 55.0 | trt 55.0 | 1.234 |
| 16 | 2 | 1.60 | 69 | 52.8 [3] | 64.7 [3] | 81.7 [3] | 94.3 [3] | 63.5 | trt 63.5 | 1.202 |
| 32 | 4 | 1.47 | 103 | 64.9 [3] | 75.9 [3] | 93.5 [3] | 103.8 [3] | 77.8 | gc 75.9 | 1.169 |
| 64 | 8 | 1.38 | 144 | 78.2 [3] | 84.6 [3] | 104.5 [3] | 113.3 [3] | 96.1 | gc 84.6 | 1.082 |
| 128 | 16 | 1.34 | 191 | 95.3 [3] | 94.6 [3] | 114.7 [3] | 123.9 [3] | 124.1 | gc 94.6 | 0.992 |
| 256 | 32 | 1.32 | 221 | 110.8 [3] | 105.5 [3] | 121.7 [3] | 130.5 [3] | 149.0 | gc 105.5 | 0.952 |
| 384 | 48 | 1.31 | 230 | 122.1 [3] | 115.8 [3] | 126.4 [3] | 133.5 [3] | 159.6 | gc 115.8 | 0.948 |
| 512 | 64 | 1.33 | 233 | 133.3 [3] | 112.9 [3] | 127.1 [3] | 133.6 [3] | 167.4 | gc 112.9 | 0.846 |

correctness (check case, real x):
- ours: rel_q 0.0324..0.0337, cos_q 0.99945..0.99949, rel_exact 3.22e-05..0.000962, cos_dense 0.99869..0.99875, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- Mega: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- fiMega: rel_q 0.0264..0.0266, cos_q 0.99965..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- trt: rel_q 0.0278..0.0282, cos_q 0.99961..0.99962, rel_exact -, cos_dense 0.99884..0.99888, nonfinite 0, det all, ranks_same all
```

### f8trG (v77 at every M (c54caf3): v77-ndev8 + LLS + XRLX + LLS grid guard, 3 reps 1f/2f/3f, trt from f8trX)

```
dsv4f_tp8 f8trG: 5 runs (f8trG_1f, f8trG_1t, f8trG_2f, f8trG_2t, f8trG_3f), hosts ['umbriel-b200-069', 'umbriel-b200-092'], ours = ['v77-ndev8@suite_b200:c54caf3']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.67 | 42 | 43.8 [3] | 57.0 [3] | 69.0 [3] | 124.1 [3] | 55.0 | trt 55.0 | 1.256 |
| 16 | 2 | 1.60 | 69 | 52.4 [3] | 64.9 [3] | 80.8 [3] | 93.2 [3] | 63.5 | trt 63.5 | 1.212 |
| 32 | 4 | 1.47 | 103 | 64.7 [3] | 75.6 [3] | 92.4 [3] | 103.0 [3] | 77.8 | gc 75.6 | 1.168 |
| 64 | 8 | 1.38 | 144 | 77.8 [3] | 84.4 [3] | 104.0 [3] | 112.9 [3] | 96.1 | gc 84.4 | 1.084 |
| 128 | 16 | 1.34 | 191 | 95.2 [3] | 94.0 [3] | 113.4 [3] | 122.9 [3] | 124.1 | gc 94.0 | 0.987 |
| 256 | 32 | 1.32 | 221 | 110.9 [3] | 105.4 [3] | 120.7 [3] | 129.9 [3] | 149.0 | gc 105.4 | 0.950 |
| 384 | 48 | 1.31 | 230 | 123.2 [3] | 114.9 [3] | 126.0 [3] | 132.7 [3] | 159.6 | gc 114.9 | 0.933 |
| 512 | 64 | 1.33 | 233 | 139.1 [3] | 112.0 [3] | 126.3 [3] | 132.7 [3] | 167.4 | gc 112.0 | 0.805 |

correctness (check case, real x):
- ours: rel_q 0.0324..0.0337, cos_q 0.99945..0.99949, rel_exact 3.22e-05..0.000962, cos_dense 0.99869..0.99875, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- Mega: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- fiMega: rel_q 0.0264..0.0266, cos_q 0.99965..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- trt: rel_q 0.0278..0.0282, cos_q 0.99961..0.99962, rel_exact -, cos_dense 0.99884..0.99888, nonfinite 0, det all, ranks_same all
```

### f8trX (v77-ndev8 + LLS + XRLX (88f4b3d); clean reps 2f + 3f (rep 1f: ours 10-15% slow at M >= 16, see the caveat))

```
dsv4f_tp8 f8trX: 4 runs (f8trX_1t, f8trX_2f, f8trX_2t, f8trX_3f), hosts ['umbriel-b200-069'], ours = ['v77-ndev8@suite_b200:88f4b3d']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.67 | 42 | 45.1 | 57.4 | 70.3 | 121.9 | 55.0 | trt 55.0 | 1.221 |
| 16 | 2 | 1.60 | 69 | 53.3 | 65.9 | 81.7 | 93.9 | 63.5 | trt 63.5 | 1.192 |
| 32 | 4 | 1.47 | 103 | 65.9 | 77.0 | 93.2 | 104.6 | 77.8 | gc 77.0 | 1.168 |
| 64 | 8 | 1.38 | 144 | 78.3 | 85.5 | 104.8 | 113.9 | 96.1 | gc 85.5 | 1.093 |
| 128 | 16 | 1.34 | 191 | 96.1 | 95.1 | 114.4 | 124.6 | 124.1 | gc 95.1 | 0.989 |
| 256 | 32 | 1.32 | 221 | 111.5 | 106.2 | 122.2 | 131.8 | 149.0 | gc 106.2 | 0.953 |
| 384 | 48 | 1.31 | 230 | 124.8 | 115.9 | 127.3 | 134.4 | 159.6 | gc 115.9 | 0.928 |
| 512 | 64 | 1.33 | 233 | 139.6 | 113.0 | 127.7 | 134.4 | 167.4 | gc 113.0 | 0.809 |

correctness (check case, real x):
- trt: rel_q 0.0278..0.0282, cos_q 0.99961..0.99962, rel_exact -, cos_dense 0.99884..0.99888, nonfinite 0, det all, ranks_same all
```

### f8trX (88f4b3d, all reps incl. the slow 1f)

```
dsv4f_tp8 f8trX: 5 runs (f8trX_1f, f8trX_1t, f8trX_2f, f8trX_2t, f8trX_3f), hosts ['umbriel-b200-069'], ours = ['v77-ndev8@suite_b200:88f4b3d']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.67 | 42 | 45.0 [3] | 57.3 [3] | 70.1 [3] | 121.4 [3] | 55.0 | trt 55.0 | 1.224 |
| 16 | 2 | 1.60 | 69 | 54.5 [3] | 65.9 [3] | 81.5 [3] | 93.6 [3] | 63.5 | trt 63.5 | 1.165 |
| 32 | 4 | 1.47 | 103 | 67.5 [3] | 76.9 [3] | 93.0 [3] | 104.8 [3] | 77.8 | gc 76.9 | 1.139 |
| 64 | 8 | 1.38 | 144 | 81.3 [3] | 85.5 [3] | 104.8 [3] | 114.1 [3] | 96.1 | gc 85.5 | 1.051 |
| 128 | 16 | 1.34 | 191 | 100.2 [3] | 95.1 [3] | 114.4 [3] | 124.4 [3] | 124.1 | gc 95.1 | 0.949 |
| 256 | 32 | 1.32 | 221 | 117.0 [3] | 106.3 [3] | 121.7 [3] | 131.5 [3] | 149.0 | gc 106.3 | 0.909 |
| 384 | 48 | 1.31 | 230 | 131.8 [3] | 115.8 [3] | 127.1 [3] | 134.2 [3] | 159.6 | gc 115.8 | 0.879 |
| 512 | 64 | 1.33 | 233 | 147.1 [3] | 114.6 [3] | 127.4 [3] | 134.2 [3] | 167.4 | gc 114.6 | 0.779 |

correctness (check case, real x):
- ours: rel_q 0.0324..0.0337, cos_q 0.99945..0.99949, rel_exact 3.22e-05..0.000962, cos_dense 0.99869..0.99875, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- Mega: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- fiMega: rel_q 0.0264..0.0266, cos_q 0.99965..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- trt: rel_q 0.0278..0.0282, cos_q 0.99961..0.99962, rel_exact -, cos_dense 0.99884..0.99888, nonfinite 0, det all, ranks_same all
```

### f8trL (v77-ndev8 + LLS)

```
dsv4f_tp8 f8trL: 4 runs (f8trL_1f, f8trL_1t, f8trL_2f, f8trL_2t), hosts ['umbriel-b200-069'], ours = ['v77-ndev8@suite_b200:06136eb']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.67 | 42 | 45.5 | 57.4 | 70.4 | 122.2 | 54.7 | trt 54.7 | 1.201 |
| 16 | 2 | 1.60 | 69 | 53.6 | 65.6 | 82.2 | 94.0 | 63.4 | trt 63.4 | 1.183 |
| 32 | 4 | 1.47 | 103 | 68.1 | 76.6 | 93.9 | 104.6 | 77.6 | gc 76.6 | 1.125 |
| 64 | 8 | 1.38 | 144 | 82.8 | 85.7 | 105.0 | 114.5 | 96.3 | gc 85.7 | 1.036 |
| 128 | 16 | 1.34 | 191 | 102.0 | 95.2 | 114.5 | 125.1 | 123.9 | gc 95.2 | 0.934 |
| 256 | 32 | 1.32 | 221 | 120.7 | 106.5 | 122.2 | 131.9 | 150.0 | gc 106.5 | 0.882 |
| 384 | 48 | 1.31 | 230 | 135.9 | 115.9 | 128.0 | 134.7 | 159.5 | gc 115.9 | 0.853 |
| 512 | 64 | 1.33 | 233 | 153.8 | 114.7 | 129.5 | 134.7 | 167.6 | gc 114.7 | 0.746 |

correctness (check case, real x):
- ours: rel_q 0.0324..0.0337, cos_q 0.99945..0.99949, rel_exact 3.22e-05..0.000962, cos_dense 0.99869..0.99875, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- Mega: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- fiMega: rel_q 0.0264..0.0266, cos_q 0.99965..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- trt: rel_q 0.0278..0.0282, cos_q 0.99961..0.99962, rel_exact -, cos_dense 0.99884..0.99888, nonfinite 0, det all, ranks_same all
```

### f8tr (v77-ndev8 (IDV, before LLS))

```
dsv4f_tp8 f8tr: 4 runs (f8tr_1f, f8tr_1t, f8tr_2f, f8tr_2t), hosts ['umbriel-b200-069'], ours = ['v77-ndev8@suite_b200:9761840', 'v77-ndev8@suite_b200:9ac74dd']
protocol b, stat us: us = mean over cases and reps (reps per arm in brackets when != 2)

| M | T | recv max/mean | U | ours | gc | Mega | fiMega | trt | best baseline | best / ours |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1.67 | 42 | 45.2 | 57.8 | 70.6 | 121.9 | 54.8 | trt 54.8 | 1.211 |
| 16 | 2 | 1.60 | 69 | 53.9 | 66.2 | 82.0 | 94.6 | 63.5 | trt 63.5 | 1.177 |
| 32 | 4 | 1.47 | 103 | 67.2 | 76.8 | 94.0 | 104.9 | 78.1 | gc 76.8 | 1.142 |
| 64 | 8 | 1.38 | 144 | 79.0 | 85.8 | 104.9 | 114.4 | 96.6 | gc 85.8 | 1.086 |
| 128 | 16 | 1.34 | 191 | 97.0 | 95.7 | 114.7 | 125.3 | 124.6 | gc 95.7 | 0.987 |
| 256 | 32 | 1.32 | 221 | 115.4 | 106.6 | 122.2 | 131.9 | 151.4 | gc 106.6 | 0.924 |
| 384 | 48 | 1.31 | 230 | 130.0 | 115.7 | 127.7 | 134.8 | 163.8 | gc 115.7 | 0.890 |
| 512 | 64 | 1.33 | 233 | 157.0 | 113.0 | 127.7 | 134.8 | 171.9 | gc 113.0 | 0.720 |

correctness (check case, real x):
- ours: rel_q 0.0324..0.0337, cos_q 0.99945..0.99949, rel_exact 3.22e-05..0.000962, cos_dense 0.99869..0.99875, nonfinite 0, det all, ranks_same all, bitwise_exact NOT all
- gc: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- Mega: rel_q 0.0265..0.0267, cos_q 0.99964..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- fiMega: rel_q 0.0264..0.0266, cos_q 0.99965..0.99965, rel_exact -, cos_dense 0.99889..0.99891, nonfinite 0, det all
- trt: rel_q 0.0278..0.0282, cos_q 0.99961..0.99962, rel_exact -, cos_dense 0.99884..0.99888, nonfinite 0, det all, ranks_same all
```

## Correctness gate of the final build (back-to-back included)

my harness `sb.py --b2b 12 --b2b-reps 4 --check 1` on the final source (log `logs/gate_f8trG.log`): per M the check case (rel vs the numerics emulation, ranks bitwise identical, deterministic) and 12-call back-to-back bursts on distinct inputs (eager with rank-skewed starts / one rank drifting / one CUDA graph of the 12 calls), every output bitwise vs its sync-separated reference on every rank (mismatching elements; 0 = clean):

```
RES M= 8 ours: rel 0.00000 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 16 ours: rel 0.00000 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 32 ours: rel 0.00000 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 64 ours: rel 0.00000 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 128 ours: rel 0.00012 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 256 ours: rel 0.00003 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 384 ours: rel 0.00003 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
RES M= 512 ours: rel 0.00006 ranks_same True det True ours b2b(K12 x4) mism eager 0 / drift 0 / graph 0 
```

## Secondary: my harness (sb.py), same trace

### c4ft8trB (my harness, v77-ndev8 + IDV (before LLS / XRLX))

```
c4ft8trB routing=trace protocol=b: us, mean of 2 reps (spread max-min)
    M           ours            trt             gc           mega best-base best/ours  recv    U
    8     44.2 ( 0.2)    54.5 ( 0.2)    56.8 ( 0.4)    69.3 ( 0.0)       trt     1.233  1.67   42
   16     52.9 ( 0.6)    61.8 ( 0.0)    63.8 ( 0.1)    80.2 ( 0.5)       trt     1.168  1.60   69
   32     64.8 ( 0.1)    76.9 ( 0.3)    75.4 ( 0.0)    92.8 ( 0.0)        gc     1.163  1.47  103
   64     78.2 ( 0.1)    96.2 ( 0.2)    83.9 ( 0.7)   104.6 ( 0.4)        gc     1.074  1.38  144
  128     95.6 ( 0.0)   122.8 ( 0.1)    93.8 ( 1.1)   113.7 ( 1.3)        gc     0.981  1.34  191
  256    114.7 ( 0.0)   148.1 ( 0.1)   105.3 ( 1.0)   121.2 ( 0.1)        gc     0.918  1.32  221
  384    129.4 ( 0.0)   164.7 ( 1.0)   114.8 ( 0.3)   126.3 ( 0.0)        gc     0.888  1.31  230
  512    146.4 ( 0.0)   174.8 ( 1.3)   111.6 ( 0.0)   126.4 ( 0.3)        gc     0.762  1.33  233
mean best-baseline/ours 1.023 over 8 M; ours wins at 4
```

## Optimizations

Kept on this cell (bitwise, 2 agreeing reps, real trace): IDV (M >= 49; TP8 A/B IDV=0 vs 1: M384 / 512 -3.2..-4.4 / -1.9..-2.9 us),
LLS (M256 / 384 / 512 -1.7 / -3.1 / -5.1 us in the clean rep; commit 06136eb), XRLX (M > 128: M256 / 384 / 512 -2.1..-1.7 /
-3.3..-2.0 / -3.3..-2.7 us; commit 88f4b3d; XTrace: tables 13.4 -> 8.8 us at M512).  Same guards and race stress as the TP4 cell
(LLS grid trap, SFBI whole-item static_assert; routed TP8 M256 / 384 / 512 up to 160k back-to-back calls per M: 0 mismatches).
- v4ft at M >= 384 (6a01f29): same-run vs v77, 2 swapped reps, TP4 M384 / 512 -4.1..-4.8 / -12.8..-13.0 us (TP8 -0.7..-2.0 / -4.1..-4.8);
  v77 stays at M <= 256 (v4ft +2..3 us there).  Gate: checks + b2b 12 every M + b2b-mix (v77 <-> v4ft switching on one stream) 0
  mismatches; stress 1500 x 8 eager + graph, 2 cases, M384 / 512 single and mixed with M64: 48k calls per rank each, 0 bad
  (results/b200/_gate_v4ft).

Refuted: as the TP4 cell (same kernel).  Where the time goes at M512 (XTrace, rank 0): prologue 9 us (tables 4.9 after XRLX,
quant 2.5), main FC1/FC2 to ~122 us (producer EMPTY waits 57 us / MMA FULL waits 23 us per CTA: the weight ring is latency-bound,
~2.5x the HBM floor), T1 ~7 us + all-reduce ~13 us (7 MiB NVLink egress + ingress per rank at ~520 GB/s: near the link bound).
11-10 budget at M512 (protocol b, PROBE_END builds, timing only): v4ft FC1-only 81.2 / + FC2 106.5 / full 133.6 us; v77 90.1 / 112.6 /
138.1.  No MMAs at all (PROBE 2): v4ft -3.6, v77 -6.7 us; no per-stage SF copies (PROBE 16): +-0.  XTrace (results/b200/_diag_xt,
xt_phase.py / xt_cta.py): the TP8 FC2 stream runs ~2.3 TB/s in both kernels, a ~0.8 us per one-stage row tile cycle co-limited by the
MMA warp (~0.53 us per tile, 0.35 without MMAs) and the FC2 epilogue (~0.45 us per tile + handoff); shared notes with B300-TP8:
/home/scratch.aichenf_wwfo/v4f_fc2_notes.md.

## Timing caveat

In some processes ours (v77 and tp8v41 alike) runs 5-15% slow at the larger M -- for one M or for the rest of the process -- while gc in the same process is normal (suite f8tr_1f / f8trL_1f / f8trX_1f, my c4ftrB_1f / k064 / k091 / k097_2).  It is not reproducible on demand: 4 repeated check-1 processes, fresh-build processes and pre-capture eager calls were all clean, and the post-timing graph output stays bitwise (post_mism 0), so it is a timing state, not a wrong-result mode.  The trigger is not understood.  The per-rep values are in each result.json (`*_1f` vs `*_2f` / `*_3f`); prefer the clean reps when comparing builds.  Node-to-node spread is ~4%; B200 event timestamps are quantized to ~2.05 us.

Reproduce (computelab, a B200 x8 hold set up with `JOB=<hold> tpmoe_suite/env/node_setup.sh sm100`):
```
cd /home/scratch.aichenf_wwfo/suite_b200
# suite harness, 2 reps (rep 1: check 1, arms fwd; rep 2: check 0, arms reversed) + trt in its own process
JOB=<hold> ./suite.sh dsv4f 8 <tag> real-trace:/home/scratch.aichenf_wwfo/routing_caps/dsv4f_real
python3 -I stable.py results/b200/dsv4f_tp8 <tag>
# my harness (paired A/B of a build change: --flags2=..., bitwise check ours2_vs_ours)
JOB=<hold> GPUS=0,1,2,3,4,5,6,7 ./sctr.sh <out> python sb.py --model dsv4f --tp 8 --ms 8 16 32 64 128 256 384 512 --arms ours gc mega --routing trace --check 1 --proto b
```
Builds are JIT (torch cpp_extension, sm_100a) into the node-local build dir on first use (~1-4 min per M build).
