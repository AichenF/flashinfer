# SM90 back-to-back race fix: H200 verification (viking-cr-196, hold 5775142, 11-09 23:05-23:40 PST)

Fix (H20-TP8, v41_h20/fix_b2b): one `named_bar_sync(1, NCONS)` right after the consumers' STAMP(P, 2), before the FC2 unit loop,
behind TP2_B2B_FIX (default 1). It closes a WAR hazard: the last FC1 epilogue's act-tile reads in the staging area (+16384..) must
finish before the first two FC2 out tiles, which have no freeo wait, overwrite that area. Ported by port_b2bfix.py into the suite
cell sources (src/: v94ex, the h123 trio, tpv41, tpv41h, tp8v41) and the full-layer sources (full/, byte-identical to H20-TP8's
fixed copies). Commit 56f1450.

Check: verify_fix.sh. Each cell's builds are run with the fix ON (default) and OFF (-DTP2_B2B_FIX=0), using b2b_check.py:
- two real-trace cases (L10s32 / L30s96), M 8..512;
- reference = one synchronized call per case;
- 16 back-to-back eager calls alternating A/B with no host sync or barrier, x 100 rounds = 1,600 calls per M;
- a graph [A,B] replayed 8x;
- all 8 (or 4) ranks; failing calls counted on rank 0.

| cell (suite adapter builds) | fix ON | fix OFF | ON == OFF bitwise (ref sha, 8 M x 2 cases) |
|---|---|---|---|
| f4  DSV4-Flash TP4 (v94ex) | PASS | PASS | yes |
| v4  DSV4.1 TP4 (tpv41 T1S=0 + tpv41 @0-32) | PASS | PASS | yes |
| f8  DSV4-Flash TP8 (h123 trio, routed / shared OFF) | PASS | **FAIL**: M384 1 / 1600, M512 1 / 1600 | yes |
| v8  DSV4.1 TP8 (tpv41 + TAIL4=0 @0-16) | PASS | **FAIL**: M64 1 / 1600, M384 2 / 1600 | yes |
| f8h DSV4-Flash TP8 hybrid (tpv41h ISPLIT=4 + h123) | PASS | PASS | yes |
| v8h DSV4.1 TP8 hybrid (tpv41h ISPLIT=4 + tpv41 TAIL4=0 @0-8) | PASS | PASS | yes |

In every failure, all ranks produced the same wrong output (ranks identical = True) and graph replay passed.

Findings:
- The race is not limited to the fused-shared-expert builds. With the fix OFF it also hits routed-only TP8 builds on H200: the h123
  trio at M384/512 and tpv41 at M64/384, at roughly 1 in 1,000-1,600 back-to-back calls.
- With the fix ON there were 0 failures across 6 cells x 8 M x 1,600 calls.
- The fix is bitwise-neutral on every cell, M and case.

(The OFF runs of v8 / f8h / v8h first crashed on a port-number bug in verify_fix.sh, 2960$p with p >= 10; they were rerun
23:23-23:40, log verify_fix_rerun.out.)
