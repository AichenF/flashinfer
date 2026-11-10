# tp41 — H20 TP8 DSV4.1-Flash (routed-only) suite adapter

Owner: agent H20-TP8.  Owner repo: .24 `/raid/sharedspace/fac/aichenf_v41` (git; mirror computelab `/home/scratch.aichenf_wwfo/v41_h20`).
`src/` = the kernel source at the commit in `src/COMMIT` (refresh with `make_src.sh`; `src/SHA256SUMS`).

- Kernel: `tp8v41.cu` with `-DTP2_MODEL=1 -DTP2_NDEV=8 -DTP2_SHARED=0` (sm_90a, torch 2.11 cu130), one launch per layer:
  in-kernel per-token E4M3 x quant + gather, FC1 stream-K (2 x 128-col items + the T4 split tail for panels <= 8 tokens), FC2 units
  (tail k-tile = one pipelined k32 slice), in-kernel LL RS + AG all-reduce.  MAXM 512.
- Build: rank 0 builds `src/tp8v41.cu` into `$SUITE_BUILD` (others wait, then load); `--ours-so SO` uses a prebuilt .so instead;
  `--opt flags=-DFOO=1:-DBAR=2` adds build flags.
- Weights: canonical MXFP4 TP slices (`ctx.tp_w()`) -> this kernel's Humming layout via `src/v41_prep.humming_from_raw`
  (humming-kernels 0.1.10; w2 trimmed to 9 k32 slices by `tp8v41.adapt_weights`).
- `xq_mode tok`; `pre_replay` zeroes `xflag`.  No `ref_partial` (the owner's exact-numerics emulation is `v41_test1.py reference()`).
- Validation (owner harness, 8 x H20 viking-prod-583): 8 ranks bitwise identical, bitwise repeatable, cos 0.99969 vs the fp32 MoE
  with per-token x quant, at M 8..512 uniform + genuine DSV4.1 L20 routing; Flash build (TP2_MODEL 0) bitwise == h123 served v78h21.

---
tpmoe_suite import (2026-11-09 21:29): `git archive 32bb910 suite_adapter` from /home/scratch.aichenf_wwfo/v41_h20 (H20-TP8), unmodified; src at 872c14c (TP2_B2B_FIX on),
verified against SHA256SUMS.  Supersedes 36a8c9a (kernels/_alt/h20_dsv41f_tp8_36a8c9a).
