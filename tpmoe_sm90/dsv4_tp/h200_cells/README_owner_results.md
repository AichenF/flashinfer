# H200 cells (agent H200) -- DSV4-Flash / DSV4.1-Flash x TP4 / TP8, routed-only, real routing

Work dir / git: /home/scratch.aichenf_wwfo/suite_h200 (local git; nothing pushed).
Kernels (src/, built by tphost.build for sm_90a, torch 2.11 cu130, nvcc 13.0, flags -O3 -gencode sm_90a -DTP2_NDEV=N -DTP2_SHARED=0 [-DTP2_MODEL=1]):
- dsv4f_tp4: v94ex = src/tp4moe2_v94ex.cu (== tpmoe_h100 pkg serving_tpmoe_tp4_h100 tp4moe2_ns.cu, sha256 bba973fb..)
- dsv4f_tp8: h123 = tp4moe2_v78p12dev.cu (62006dc5..) + v94p10dev.cu (a54e3bdc..) @M64-127 + v94p11dev.cu (84fd8f91..) @M128-341
- dsv41f_tp4 / tp8: src/tpv41.cu = port_v41_tp4.py(src/tp8v41.cu) (tp8v41.cu = copy of v41_h20/tp8v41.cu, the H20-TP8 agent's port_v41.py
  output from h123's v78h21); TP8 build bitwise == tp8v41.cu build; TP4: 576 = 4 items + 64-col tail item, NKT2 5, RS owner (rtile+t)%4.
Suite adapters: cells/h200_<model>_tp<N>/adapter.py (shared cells/h2cell.py).  Own harness: h2bench.py (cmp3/v41_bench generalized to N).
Per-cell dirs: logs (raw/), table.md, README.md (build, commit, command).

## Status (2026-11-09 evening)
- Headline (real-trace v2, 8x H200 viking-prod-322 full NV18 mesh, suite harness, protocol b, 2-3 reps): ours beats gc-moe EP and
  flashinfer MegaMoE EP at every M in all 4 cells (tables in <cell>/table.md).
- TP8-hybrid variant (separate row, coordinator decision): 2 expert groups x TP4 slices, tokens replicated, 8-way in-kernel AR
  (src/tpv41h.cu = port_isplit.py(tpv41.cu); adapters cells/h200_dsv41f_tp8h, cells/h200_dsv4f_tp8h).
- Correctness gate: suite real-x check (rel_q 0.025, cos_dense 0.999, ranks_same, det) + back-to-back race check b2b_check.py
  (16 eager alternating calls without sync + graph [A,B] x8; ALL PASS for 4 cells + 2 hybrids, M8..512).
- h200-nvl (4u8g-gen-0277): 8x H200 NVL, two 4-GPU NV6 meshes, P2P only within a group -> TP4 cells only (results/raw/nvl).
