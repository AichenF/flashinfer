| T | M | recv max/mean (c3/c11) | h123full | gcfull | fimxfull | fimxfull2 | gcfull/TP | fimxfull/TP | fimxfull2/TP | best EP/TP | rep spread % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 8 | 1.33/1.50 | 95.4 | 156.8 | 160.8 | 160.3 | 1.64 | 1.69 | 1.68 | 1.64 (gcfull) | 0.7 |
| 4 | 16 | 1.21/1.38 | 137.4 | 200.9 | 218.6 | 219.5 | 1.46 | 1.59 | 1.60 | 1.46 (gcfull) | 0.5 |
| 8 | 32 | 1.33/1.17 | 180.1 | 225.1 | 255.8 | 254.0 | 1.25 | 1.42 | 1.41 | 1.25 (gcfull) | 0.6 |
| 16 | 64 | 1.04/1.10 | 244.4 | 288.4 | 343.8 | 343.2 | 1.18 | 1.41 | 1.40 | 1.18 (gcfull) | 0.3 |
| 32 | 128 | 1.03/1.07 | 293.3 | 345.4 | 415.7 | 413.9 | 1.18 | 1.42 | 1.41 | 1.18 (gcfull) | 0.6 |
| 64 | 256 | 1.03/1.04 | 363.6 | 426.7 | 499.6 | 499.7 | 1.17 | 1.37 | 1.37 | 1.17 (gcfull) | 0.8 |
| 96 | 384 | 1.10/1.05 | 421.1 | 520.3 | 569.1 | 570.6 | 1.24 | 1.35 | 1.35 | 1.24 (gcfull) | 0.4 |

runs: 28 TP points; h123full (repeat_bitwise, ranks_identical, nonfinite): [(True, True, 0)]
gcfull (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
fimxfull (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
fimxfull2 (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
cross rel_rms fimxfull~fimxfull2: 0.0 .. 0.0
cross rel_rms gcfull~fimxfull: 0.0354 .. 0.036
cross rel_rms h123full~fimxfull: 0.0642 .. 0.0653
cross rel_rms h123full~gcfull: 0.0629 .. 0.0641
EP shared expert: checkpoint FP8 128x128 block scales as-is | check (M, gemm1 rel, gemm2 rel, vs ckpt rel): (('1', 0.0017, 0.0017, 0.021), ('5', 0.0017, 0.0017, 0.026), ('32', 0.0017, 0.0017, 0.022))
