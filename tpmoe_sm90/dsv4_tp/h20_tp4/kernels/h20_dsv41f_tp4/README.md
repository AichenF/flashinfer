# h20_dsv41f_tp4 — tp4v41 (H20 TP4, DeepSeek-V4.1-Flash, routed-only)

- Source: `suite_h20b/src/tp4v41.cu` = `port_tp4v41.py` applied to H20-TP8's `tp8v41.cu` (v41_h20 git 650c980; h123 lineage
  `tp4moe2_v78p12dev.cu` via `port_v41.py`). TP4 changes: per-rank I = 576 = 4 items + a 64-column tail item (NQG 5; tail w13
  = [gate rows 512..575 | OOB | up rows 512..575 | OOB] from its own maps), FC2 K 576 = 5 k-steps (the last k-tile's slices
  18, 19 TMA zero-fill), reduce-scatter owner (rtile + t) % 4 with 5 slots; T4 split tail stays TP8-only.
- Build: `-O3 -gencode=arch=compute_90a,code=sm_90a -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -DTP2_MODEL=1 -DTP2_NDEV=4 -DTP2_SHARED=0`
  (nvcc 13.0.88, the compiler of the served h123 binaries); 96 regs, 160 B stack.
- Host: `src/tp8v41.py` (RankState / run, new ABI with tail weights). Weights: canonical MXFP4 slices -> Humming layout via
  `src/v41_prep.py` (main stacks [512, 512] K 5120, tail [64, 64], w2 K 576).
- Validation: 1-GPU rank-local vs a full numerics emulation (v41_test1.py --ndev 4) rel 3-5e-3 on all 4 ranks, M 1..512
  (same level as the TP8 port); 4-GPU: deterministic, all ranks bitwise identical, rel 0.0256 vs the per-token-quant fp32 ref.
- Cross-check suite harness (7658da0, run in cmp3env) vs suite_h20b/bench/h20b_bench.py, real cap_v41_l20, protocol b:
  agree within 1% at every M (e.g. M8 75.0/76.0 vs 75.6/75.8, M512 667.6/667.3 vs 668.1/667.0 us).

## Kept optimization 1 (11-09): tp4v41c = T2 split tail + warp-uniform descriptor offset (default src)
- `port_t2.py` (tp4v41 -> tp4v41b): the tail item's two real 64-row blocks (gate 512..575 | up 512..575) split 2 ways over K across the
  4 warpgroups for panels of width <= 16 (instead of 2 warpgroups multiplying TMA zero-filled out-of-bounds blocks); weights `w1t4/s1t4/f1t4`
  = [E, 256, H/2] Humming stacks [128, 128] (v41_prep.py, nt == 64).
- tp4v41b alone was neutral: ncu SourceCounters showed QGMMA -6.7% (the expected tail k-steps) but IMAD +13% and R2UR in the hot loop:
  the per-warpgroup B-descriptor offset `(L.w & 1) * ...` was not provably warp-uniform -> ptxas kept the k-loop descriptors in vector
  registers + R2UR per wgmma.  tp4v41c broadcasts it with `__shfl_sync(.., 0)` (as CUTLASS canonical_warp_group_idx): executed
  instructions -6.1% at M128 (2.407e8 vs 2.562e8), stack 176 B, no C75xx.
- Graph bench (586, in-process vs tp4v41, 2 reps): real headline v2 -2.0% (M8), -3.6..-4.4% (M16..512); uniform -3.4..-4.9%.
  Outputs bitwise == tp4v41 on all 4 ranks at every M / case tested (T2 changes only the fp32 association of the tail's K halves;
  identical after BF16 / e4m3 rounding here); 1-GPU rank-local vs emulation: ALL_OK on ranks 0-3, M 1..512 (rel 3-5e-3).

## Kept optimization 2 (11-09): tp4v41e = tp4v41c + FC2 half k-tile first (default src)
- FC2 K = 576 = 4.5 k-tiles: the last k-tile's slices 18, 19 are TMA zero fill.  The 2-real-slice k-step as the LAST step (kstep2 /
  KPipeH, 11-09) hit ptxas C7512 (all wgmma serialized, +3..5%); as the FIRST step (producer k-tile order 4,0,1,2,3, `kpipe2h.cuh`
  KPipeH2<N, true> = KPipe<N, true> truncated after slice 1, TP2_FC2_HFIRST) ptxas is clean (no C75xx, 96 regs, 176 B stack).
- Graph bench (586, in-process vs tp4v41c, 2 reps each): real headline v2 -0.5..-1.2% at every M, uniform -0.3..-1.2%, every rep agrees;
  outputs bitwise == tp4v41c (all 4 ranks, 56 trace cases + 8 uniform M); 1-GPU emulation ALL_OK on ranks 0-3 (M 1..512).
