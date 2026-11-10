# H200 full layer TP vs EP on real inputs (viking-cr-196, hold 5775142, 11-10 00:49-01:01 PST)

Bench: full/full_bench2.py = H20-TP8 fimega_full v41_bench.py (sha d37920de44768685) + patch_full2.py (env paths, b2b8 stress, B2B_ROUNDS=200). All TP builds have the SM90 b2b race fix (TP2_B2B_FIX=1). Cases 3 / 11 (steps 36 / 100) x reps r1 / r2 (swapped order); per-run protocol b; table = mean over reps then cases.

## DSV4.1-Flash TP8 (tp41full = tp8v41 SV: fused 384 gate + SH32 shared + AR)

| T | M | recv max/mean (c3/c11) | U | tp41full us (reps) | gcfull us (reps) | fimxfull us (reps) | fimxfull2 us (reps) | gcfull/tp41full | fimxfull/tp41full | fimxfull2/tp41full | best EP-full / TP-full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 2.00/1.67 | 45/40 | 88.3 (88.2 / 88.4) | 120.2 (120.0 / 120.4) | 156.3 (156.3 / 156.4) | 155.1 (155.2 / 155.1) | 1.36 | 1.77 | 1.76 | **1.36** |
| 2 | 16 | 1.58/1.58 | 68/73 | 108.2 (108.1 / 108.4) | 134.6 (134.6 / 134.6) | 174.6 (174.7 / 174.5) | 172.5 (172.5 / 172.5) | 1.24 | 1.61 | 1.59 | **1.24** |
| 4 | 32 | 1.33/1.29 | 108/115 | 131.7 (131.8 / 131.6) | 173.4 (173.2 / 173.5) | 202.6 (202.8 / 202.4) | 199.9 (200.2 / 199.5) | 1.32 | 1.54 | 1.52 | **1.32** |
| 8 | 64 | 1.38/1.35 | 165/163 | 171.3 (171.1 / 171.4) | 254.6 (254.4 / 254.8) | 235.4 (235.4 / 235.4) | 232.3 (232.5 / 232.2) | 1.49 | 1.37 | 1.36 | **1.36** |
| 16 | 128 | 1.39/1.44 | 226/226 | 223.0 (222.8 / 223.1) | 370.7 (370.7 / 370.6) | 314.7 (314.9 / 314.5) | 312.7 (312.7 / 312.6) | 1.66 | 1.41 | 1.40 | **1.40** |
| 32 | 256 | 1.39/1.39 | 278/294 | 286.5 (286.5 / 286.5) | 403.9 (404.2 / 403.6) | 368.9 (369.2 / 368.6) | 364.3 (364.9 / 363.7) | 1.41 | 1.29 | 1.27 | **1.27** |
| 48 | 384 | 1.45/1.44 | 306/321 | 331.3 (331.1 / 331.4) | 567.3 (566.7 / 568.0) | 398.2 (398.4 / 397.9) | 394.1 (394.1 / 394.0) | 1.71 | 1.20 | 1.19 | **1.19** |
| 64 | 512 | 1.53/1.39 | 318/332 | 371.9 (372.0 / 371.8) | 793.3 (793.4 / 793.3) | 418.1 (418.1 / 418.1) | 413.3 (413.7 / 412.9) | 2.13 | 1.12 | 1.11 | **1.11** |

- gcfull: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_tp41 min 1.0
- tp41full: repeat_bitwise [True], ranks_identical [True], b2b8_bitwise [True], b2b8_ranks_identical [True], b2b8 calls total 51200, nonfinite max 0
- fimxfull: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_tp41full min 1.0
- fimxfull2: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_tp41full min 1.0
cross-arm rel_rms: tp41full~gcfull 0.0617..0.0627; tp41full~fimxfull 0.0628..0.0641; gcfull~fimxfull 0.0351..0.0354; fimxfull~fimxfull2 0.0000..0.0000

## DSV4-Flash TP8 (h123full = h123 sh builds v78h21 / v94h13 @64-127 / v94h21 @128-341)

| T | M | recv max/mean (c3/c11) | U | h123full us (reps) | gcfull us (reps) | fimxfull us (reps) | fimxfull2 us (reps) | h123 us (reps) | gcfull/h123full | fimxfull/h123full | fimxfull2/h123full | best EP-full / TP-full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 1.67/2.33 | 38/44 | 73.0 (73.0 / 73.0) | 95.3 (95.5 / 95.1) | 116.4 (117.0 / 115.8) | 115.0 (115.2 / 114.8) | 55.9 (55.8 / 56.0) | 1.31 | 1.60 | 1.58 | **1.31** |
| 2 | 16 | 1.42/1.75 | 72/68 | 87.6 (87.7 / 87.5) | 108.3 (108.6 / 108.0) | 132.9 (133.0 / 132.9) | 131.4 (131.8 / 131.1) | 68.5 (68.7 / 68.4) | 1.24 | 1.52 | 1.50 | **1.24** |
| 4 | 32 | 1.58/1.29 | 98/110 | 103.0 (103.1 / 102.9) | 123.0 (123.2 / 122.7) | 150.2 (150.8 / 149.6) | 149.2 (149.6 / 148.8) | 87.8 (88.0 / 87.7) | 1.19 | 1.46 | 1.45 | **1.19** |
| 8 | 64 | 1.46/1.19 | 147/158 | 122.6 (122.8 / 122.5) | 155.7 (155.7 / 155.6) | 186.0 (186.4 / 185.7) | 183.4 (183.2 / 183.7) | 110.4 (110.6 / 110.3) | 1.27 | 1.52 | 1.50 | **1.27** |
| 16 | 128 | 1.29/1.30 | 172/192 | 141.2 (141.2 / 141.2) | 239.9 (240.1 / 239.6) | 206.2 (206.1 / 206.2) | 203.5 (203.4 / 203.6) | 129.5 (129.5 / 129.5) | 1.70 | 1.46 | 1.44 | **1.44** |
| 32 | 256 | 1.30/1.26 | 213/214 | 168.5 (168.4 / 168.6) | 229.5 (229.8 / 229.3) | 222.5 (222.1 / 222.9) | 218.0 (218.1 / 217.9) | 152.6 (152.7 / 152.5) | 1.36 | 1.32 | 1.29 | **1.29** |
| 48 | 384 | 1.34/1.26 | 221/230 | 191.6 (191.6 / 191.5) | 285.8 (285.9 / 285.6) | 245.8 (246.1 / 245.6) | 245.1 (245.1 / 245.0) | 172.9 (172.9 / 172.9) | 1.49 | 1.28 | 1.28 | **1.28** |
| 64 | 512 | 1.32/1.21 | 220/231 | 213.6 (213.5 / 213.7) | 360.9 (361.1 / 360.8) | 256.6 (256.6 / 256.7) | 256.9 (256.8 / 257.0) | 193.7 (193.6 / 193.7) | 1.69 | 1.20 | 1.20 | **1.20** |

- h123: repeat_bitwise [True], ranks_identical [True], b2b8_bitwise [True], b2b8_ranks_identical [True], b2b8 calls total 51200, nonfinite max 0, cos_vs_ref_q 0.99968..0.99969
- gcfull: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_tp min 1.0
- h123full: repeat_bitwise [True], ranks_identical [True], b2b8_bitwise [True], b2b8_ranks_identical [True], b2b8 calls total 51200, nonfinite max 0
- fimxfull: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_h123full min 1.0
- fimxfull2: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_h123full min 1.0
cross-arm rel_rms: h123full~gcfull 0.0623..0.0642; h123full~fimxfull 0.0636..0.0654; gcfull~fimxfull 0.0353..0.0359; fimxfull~fimxfull2 0.0000..0.0000

## DSV4-Flash TP4 (h123full = v94ex sh build, NDEV 4; EP4)

| T | M | recv max/mean (c3/c11) | U | h123full us (reps) | gcfull us (reps) | fimxfull us (reps) | fimxfull2 us (reps) | h123 us (reps) | gcfull/h123full | fimxfull/h123full | fimxfull2/h123full | best EP-full / TP-full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 8 | 1.33/1.50 | 38/44 | 86.1 (86.1 / 86.2) | 108.5 (108.3 / 108.7) | 131.8 (131.6 / 132.0) | 129.9 (129.7 / 130.2) | 68.7 (68.6 / 68.8) | 1.26 | 1.53 | 1.51 | **1.26** |
| 4 | 16 | 1.21/1.38 | 72/68 | 110.7 (110.5 / 110.9) | 149.2 (148.9 / 149.5) | 177.2 (176.5 / 177.9) | 176.9 (176.6 / 177.2) | 93.7 (93.5 / 93.8) | 1.35 | 1.60 | 1.60 | **1.35** |
| 8 | 32 | 1.33/1.17 | 98/110 | 147.1 (146.7 / 147.4) | 174.6 (174.3 / 174.9) | 198.1 (198.2 / 198.0) | 195.4 (195.2 / 195.6) | 126.0 (125.8 / 126.1) | 1.19 | 1.35 | 1.33 | **1.19** |
| 16 | 64 | 1.04/1.10 | 147/158 | 185.6 (185.6 / 185.7) | 229.4 (229.3 / 229.4) | 264.6 (264.6 / 264.6) | 263.2 (263.1 / 263.3) | 166.5 (166.3 / 166.6) | 1.24 | 1.43 | 1.42 | **1.24** |
| 32 | 128 | 1.03/1.07 | 172/192 | 216.5 (216.3 / 216.7) | 251.1 (251.1 / 251.0) | 283.1 (283.0 / 283.2) | 282.8 (282.8 / 282.7) | 196.4 (196.3 / 196.5) | 1.16 | 1.31 | 1.31 | **1.16** |
| 64 | 256 | 1.03/1.04 | 213/214 | 260.3 (260.5 / 260.2) | 314.5 (315.0 / 314.0) | 336.8 (337.0 / 336.5) | 337.2 (337.8 / 336.7) | 233.2 (233.2 / 233.2) | 1.21 | 1.29 | 1.30 | **1.21** |
| 96 | 384 | 1.10/1.05 | 221/230 | 294.2 (294.1 / 294.3) | 376.3 (376.0 / 376.6) | 372.8 (372.7 / 372.8) | 371.3 (371.2 / 371.5) | 259.1 (259.0 / 259.2) | 1.28 | 1.27 | 1.26 | **1.26** |
| 128 | 512 | 1.08/1.04 | 220/231 | 326.8 (326.8 / 326.7) | 454.7 (455.2 / 454.2) | 407.4 (407.9 / 406.9) | 408.3 (408.5 / 408.1) | 284.2 (284.2 / 284.2) | 1.39 | 1.25 | 1.25 | **1.25** |

- h123: repeat_bitwise [True], ranks_identical [True], b2b8_bitwise [True], b2b8_ranks_identical [True], b2b8 calls total 51200, nonfinite max 0, cos_vs_ref_q 0.99967..0.99967
- gcfull: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_tp min 1.0
- h123full: repeat_bitwise [True], ranks_identical [True], b2b8_bitwise [True], b2b8_ranks_identical [True], b2b8 calls total 51200, nonfinite max 0
- fimxfull: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_h123full min 1.0
- fimxfull2: repeat_bitwise [True], nonfinite max 0, routing_rows_equal_to_h123full min 1.0
cross-arm rel_rms: h123full~gcfull 0.0628..0.0645; h123full~fimxfull 0.0639..0.0657; gcfull~fimxfull 0.0353..0.0359; fimxfull~fimxfull2 0.0000..0.0000
