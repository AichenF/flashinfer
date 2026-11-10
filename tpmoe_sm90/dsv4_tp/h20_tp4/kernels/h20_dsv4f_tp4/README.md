# h20_dsv4f_tp4 — v94ex at NDEV 4 (H20 TP4, DeepSeek-V4-Flash, routed-only ns build)

- Source: `suite_h20b/src/tp4moe2_v94ex.cu` (= tpmoe_h100 git 5c3aa17: v94p11dev (h123 lineage) + LL epoch-parity double buffers
  + TP2_SH_FIRST 0 + TP2_SMALL_MAXMT 128). Bitwise == v28 (= served TP4 v27 ns numerics) on all 4 ranks at every M (h20b bench
  cross-check, real routing M 8..256 and uniform M 8..512); v94p11 / v78p12 at NDEV 4 are also bitwise equal and within ~1 us.
- Build: `-O3 -gencode=arch=compute_90a,code=sm_90a -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -DTP2_SHARED=0` (NDEV 4 default), nvcc 13.0.88.
- Host: `src/tp4moe2_par.py` (RS / AG buffers x2: parity halves). Weights: suite humming_prep.
- vs the v26/v28 served kernel on H20 (same process, real routing): -1..-5 us (M16 95.8 -> 90.6, M32 119.6 -> 115.1, M256 263.6 -> 258.0).
- Cross-check suite harness vs h20b_bench (real cap_l20): M8 68.7 vs 69.4, M64 152.5 vs 152.2, M256 258.6 vs 258.0 us.
