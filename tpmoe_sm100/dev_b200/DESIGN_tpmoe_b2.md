# tpmoe_b2: DeepSeek-V4-Flash TP4 fused MoE decode kernel for B200 (sm_100a)

Port of tp4moe2 (H20/H100, wgmma) to B200. The skeleton is tpmoe's: hook, fused gate, panels, stream-K FC1, FC2 queue, T1,
LL all-reduce. The compute core is new tcgen05 code, after the B200 MoE kernels of the dsv41 effort (read-only reference).

## Baseline (what the hook replaces)
`--moe-runner-backend flashinfer_mxfp4` (SGLang B200 TP4 recipe):
1. router GEMM (dsv3_router_gemm for M <= 4, else cuBLAS fp32);
2. moe_fused_gate (Triton, sqrtsoftplus, renorm);
3. PackTopkIds (bf16 weights) + mxfp8_quantize (cute-dsl) + trtllm_fp4_block_scale_routed_moe (do_finalize, bf16 out);
4. DeepGEMM FP8 shared expert on the alt stream (UE8M0 block scales);
5. shared.add_(routed, alpha=1.5);
6. CustomAllReduceV2.

## Numerics (production scheme; not bitwise: other accumulation order)
- x: MXFP8 per 32 (e4m3, UE8M0; convention checked against flashinfer mxfp8_quantize).
- FC1: MXFP4 w13 x MXFP8 x, fp32 accumulation over K 4096 in TMEM.
- act = up * gate * sigmoid(gate), with gate <= 10 and |up| <= 10; h = MXFP8 per 32 of act.
- FC2: fp32, then bf16 per (token, slot).
- T1: r = bf16(sum_k bf16(w_k) * o_k) in k order, then y = bf16(shared + 1.5 r).
- All-reduce: fp32 sum in rank order, bf16.

## Weights: SGLang's Mxfp4FlashinferTrtllmMoEMethod tensors, read in place
- w13 u8 [256, 1024, 2048]: 8 tiles of 128 rows; tile T = 64 gate + 64 up rows of h columns 64T..64T+63 (lane tables).
- w13_sf [256, 1024, 128]: block_scale_interleave = the tcgen05 SFA image, 512 B per (tile, 128-k chunk).
- w2 u8 [256, 4096, 256]: 32 row tiles of 128 output rows, permuted inside a tile (FC2 lane table); K 512 = 4 chunks.
- w2_sf [256, 4096, 16]: 512 B per (row tile, chunk).
- Lane tables are probed by b2_umma_test.py; the kernel takes them as input.

## Kernel structure (one persistent CTA per SM, 8 warps)
| warp | role |
|---|---|
| w0 lane 0 | producer: A by TMA (16U4_ALIGN16B 3-D map) + SFA bulk; ring of work descriptors |
| w1 | TMEM alloc; MMA issuer (converged warp, elect.sync) |
| w2..w5 | epilogue (TMEM lane quadrant warp & 3) |
| w6 | B operand: FC1 x rows -> SW128 smem + SFB; FC2 h image by bulk copy after the panel flag |
| w7 lane 0 | scale copier (tcgen05.cp 32x128b.warpx4 SFA / SFB -> TMEM) |

- Prologue (all warps): gate, tables, x MXFP8 quantization; PDL wait first.
- FC1:
  - Items: (panel <= 32 tokens, w13 tile T) x 32 chunks; stage = 2 chunks.
  - Stream-K over the grid. A CTA's first piece may be a non-owner partial (fp32 to part[cta]); the owner (chunk 0) adds the partials in CTA order.
  - Epilogue: TMEM -> fp32 exch smem -> SwiGLU (lane tables) -> MXFP8 h image + SFB image (global, SW128) -> pan_cnt[p]++.
- FC2:
  - Units (row tile r, panel p) from a global queue (row-tile major); stage = 2 chunks; B = h of the panel after pan_cnt[p] == 8.
  - Epilogue: bf16 per (token, slot) -> outp; the store also counts row tiles for T1.
- T1 + all-reduce: tpmoe's tail (T1 queue per 256-row tile, one-shot LL for small M, RS + AG otherwise, epoch parity buffers).
- SMEM:
  - ring: 4 stages x 43 KB (A 32 KB unpacked e2m1, B <= 8 KB, SFA 1 KB, SFB 512 B);
  - exch 16 KB; tables ~30 KB.
- TMEM (512 columns): accumulators 4 x 32 columns at 0; scale factors at 256 + 32 per stage.

## Unit test (umma1, 10-30 umb-b200-237): PASS
- tcgen05 kind::mxf8f6f4 block32, M128, N = 8 / 16 / 24 / 32: exact on SGLang's in-place w13 (8 tiles) and w2 (32 tiles), experts 0 / 77 / 255.
- SFB for N <= 32: token n at lane n.
- Lane tables: identical to dsv41's SGL_GATE_LANE / SGL_UP_LANE / SGL_FC2_COL, the same in every tile and expert.
  - Gate lane = up lane + 8, always in the same warp: up lanes are (l & 8) == 0.
  - In quadrant q, up lane l holds h column 16q + 2(l & 7) + ((l >> 4) & 1). One warp therefore holds 16 h columns = one 16-B granule per token.
  - FC2 row of lane L: SGL_FC2_COL[L]. Lanes 32w..32w+31 cover rows 32w..32w+31.
