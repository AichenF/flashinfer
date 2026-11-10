| T | M | recv max/mean (c3/c11) | tp41full | gcfull | fimxfull | fimxfull2 | gcfull/TP | fimxfull/TP | fimxfull2/TP | best EP/TP | rep spread % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 2.00/1.67 | 94.8 | 168.6 | 169.6 | 169.7 | 1.78 | 1.79 | 1.79 | 1.78 (gcfull) | 0.4 |
| 2 | 16 | 1.58/1.58 | 130.4 | 193.6 | 200.9 | 199.2 | 1.48 | 1.54 | 1.53 | 1.48 (gcfull) | 0.6 |
| 4 | 32 | 1.33/1.29 | 170.5 | 262.4 | 248.9 | 246.3 | 1.54 | 1.46 | 1.44 | 1.44 (fimxfull2) | 1.0 |
| 8 | 64 | 1.38/1.35 | 231.5 | 425.7 | 289.2 | 284.2 | 1.84 | 1.25 | 1.23 | 1.23 (fimxfull2) | 1.4 |
| 16 | 128 | 1.39/1.44 | 309.6 | 631.4 | 396.9 | 391.0 | 2.04 | 1.28 | 1.26 | 1.26 (fimxfull2) | 2.5 |
| 32 | 256 | 1.39/1.39 | 409.1 | 562.6 | 556.5 | 552.1 | 1.38 | 1.36 | 1.35 | 1.35 (fimxfull2) | 1.0 |
| 48 | 384 | 1.45/1.44 | 477.5 | 757.1 | 593.9 | 588.4 | 1.59 | 1.24 | 1.23 | 1.23 (fimxfull2) | 0.7 |
| 64 | 512 | 1.53/1.39 | 541.0 | 1045.2 | 632.2 | 630.9 | 1.93 | 1.17 | 1.17 | 1.17 (fimxfull2) | 1.0 |

runs: 32 TP points; tp41full (repeat_bitwise, ranks_identical, nonfinite): [(True, True, 0)]
gcfull (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
fimxfull (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
fimxfull2 (repeat_bitwise, nonfinite, routing rows equal to TP gate): [(True, 0, 1.0)]
cross rel_rms fimxfull~fimxfull2: 0.0 .. 0.0
cross rel_rms gcfull~fimxfull: 0.0353 .. 0.0356
cross rel_rms tp41full~fimxfull: 0.0629 .. 0.0639
cross rel_rms tp41full~gcfull: 0.0619 .. 0.0627
EP shared expert: checkpoint FP8 with 32x32 scales requantized to 128x128 power-of-two block scales | check (M, gemm1 rel, gemm2 rel, vs ckpt rel): (('1', 0.0017, 0.0017, 0.022), ('5', 0.0017, 0.0016, 0.022), ('32', 0.0017, 0.0017, 0.021))
