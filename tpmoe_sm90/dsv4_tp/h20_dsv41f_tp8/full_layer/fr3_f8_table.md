| T | M | recv max/mean (c3/c11) | h123full | gcfull | fimxfull | fimxfull2 | gcfull/TP | fimxfull/TP | fimxfull2/TP | best EP/TP | rep spread % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 1.67/2.33 | 72.3 | 137.0 | 135.4 | 133.8 | 1.89 | 1.87 | 1.85 | 1.85 (fimxfull2) | 0.5 |
| 2 | 16 | 1.42/1.75 | 90.6 | 156.9 | 164.4 | 163.4 | 1.73 | 1.82 | 1.80 | 1.73 (gcfull) | 0.6 |
| 4 | 32 | 1.58/1.29 | 112.3 | 177.6 | 179.3 | 178.4 | 1.58 | 1.60 | 1.59 | 1.58 (gcfull) | 0.9 |
| 8 | 64 | 1.46/1.19 | 146.7 | 237.2 | 230.6 | 228.0 | 1.62 | 1.57 | 1.55 | 1.55 (fimxfull2) | 0.5 |
| 16 | 128 | 1.29/1.30 | 175.0 | 412.8 | 265.1 | 263.9 | 2.36 | 1.51 | 1.51 | 1.51 (fimxfull2) | 3.3 |
| 32 | 256 | 1.30/1.26 | 216.7 | 296.5 | 309.4 | 306.5 | 1.37 | 1.43 | 1.41 | 1.37 (gcfull) | 1.1 |
| 48 | 384 | 1.34/1.26 | 252.1 | 385.0 | 340.9 | 341.1 | 1.53 | 1.35 | 1.35 | 1.35 (fimxfull) | 2.6 |

runs: 28 TP points; h123full (repeat_bitwise, ranks_identical, nonfinite): [(True, True, 0)]
gcfull (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
fimxfull (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
fimxfull2 (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
cross rel_rms fimxfull~fimxfull2: 0.0 .. 0.0
cross rel_rms gcfull~fimxfull: 0.0354 .. 0.036
cross rel_rms h123full~fimxfull: 0.0638 .. 0.0651
cross rel_rms h123full~gcfull: 0.0624 .. 0.0639
EP shared expert: checkpoint FP8 128x128 block scales as-is | check (M, gemm1 rel, gemm2 rel, vs ckpt rel): (('1', 0.0017, 0.0017, 0.021), ('5', 0.0017, 0.0017, 0.026), ('32', 0.0017, 0.0017, 0.022))
