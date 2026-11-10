// tpmoe_b2: TP4 fused MoE decode kernel for B200 (sm_100a), DeepSeek-V4-Flash routed experts (hidden 4096, per-rank intermediate 512,
// 256 experts, top-k <= 8, M <= 512), SGLang's flashinfer_mxfp4 (trtllm-gen) expert tensors used in place.  The B200 port of tp4moe2
// (H20 / H100): same skeleton -- fused gate, routing tables with token panels of <= 32 entries, stream-K FC1, FC2 unit queue, T1 and the
// LL all-reduce in the same launch -- with a tcgen05 core: kind::mxf8f6f4 block-scaled MMA (MXFP4 weights x MXFP8 activations, UE8M0 per
// 32), fp32 accumulators in tensor memory, one MMA-issuing warp.  Design: design/DESIGN_tpmoe_b2.md.
//
// Numerics (the production flashinfer_mxfp4 scheme; not bitwise -- trtllm-gen sums in another order):
//   x  -> MXFP8 (e4m3, one UE8M0 scale per 32: 2^ceil(log2(amax / 448)), = flashinfer mxfp8_quantize)   FC1 fp32 (TMEM, K 4096)
//   act = up * gate * sigmoid(gate), gate <= 10, |up| <= 10 (fp32) -> h = MXFP8 per 32 with the OCP rule 2^(floor(log2 amax) - 8)
//   (trtllm-gen's FC1 output quantization, measured: b2_refcheck)                 FC2 fp32 -> bf16 per (token, slot)
//   T1: r = bf16(sum_k bf16(w_k) * o_k) (k order), y = bf16(shared + 1.5 r)     all-reduce: rank-ordered fp32 sum -> bf16
//
// One persistent CTA per SM, 8 warps:  w0 lane 0 producer (weights by TMA + SFA; work descriptors on the ring) | w1 TMEM + MMA |
// w2..w5 epilogue (TMEM lane quadrant warp & 3) | w6 B operand (FC1: x rows -> SW128 smem + SFB; FC2: h image by bulk copy) |
// w7 lane 0 scale copier (tcgen05.cp -> TMEM).  Prologue and tail use all 256 threads.
#ifndef B2_NO_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

#ifndef TPB2_NS
#define TPB2_NS tpb2_v23
#endif
#ifndef TPB2_SMALLM
#define TPB2_SMALLM 8   // fused gate: M <= this -> every CTA gates every token itself (no grid-wide routing barrier)
#endif
#ifndef TPB2_SHCOST
#define TPB2_SHCOST 2   // stream-K cost of a shared FC1 stage (2 x 16 KB e4m3) in routed stages (16 KB e2m1)
#endif
#ifndef TPB2_MERGE_B
#define TPB2_MERGE_B 1  // split-K merge: contributors' partial loads in flight together (routed; shared: half)
#endif
#ifndef TPB2_RR
#define TPB2_RR 1   // FC1 work list round-robin over the CTAs (no routed split-K) when there are >= 2 G routed items and the shared quarter
                    // pieces fit one wave: panels complete in waves, CTAs leaving FC1 early run FC2 of ready panels (0: stream-K always)
#endif
#ifndef TPB2_RR_MIN4
#define TPB2_RR_MIN4 0   // rr when routed items >= TPB2_RR_MIN4 / 4 waves or < TPB2_RR_LO4 / 4 waves, else stream-K.  0 = rr at every M:
#endif                    // with SHW1 (v42) stream-K's split-item merges in the streaming phase cost more than rr's last-wave imbalance
                          // (paired v43: M6 -1.6/-2.0%, M8 -8.2/-4.9%, M12 -6.8%, M16 -9.9%; v37 had stream-K ahead at M6-16 by 0.6-4.9%)
#ifndef TPB2_RR_LO4
#define TPB2_RR_LO4 5
#endif
#ifndef TPB2_MPAIR
#define TPB2_MPAIR 0   // MMA warp: two ring stages (16 MMAs) per issue round (measured: no gain -- the stream is data-bound)
#endif
#ifndef TPB2_PF
#define TPB2_PF 0   // L2 prefetch distance of the FC1 weight stream in ring stages (0: off); FC2 weights of an FC1 item's expert slice are
                     // prefetched when the item starts.  (PROBE=2 measured: without any MMA the FC1 stream still runs at ~6 TB/s -- bound by
                     // the 4 x 16 KB of weights in flight per SM, not by the tensor core)
#endif
#ifndef TPB2_PF2
#define TPB2_PF2 0   // with TPB2_PF: also prefetch each FC1 item's FC2 weight slice (measured: evicted before use, the FC2 weights read twice)
#endif
#ifndef TPB2_SFCP_MMA
#define TPB2_SFCP_MMA 1   // the MMA warp issues a stage's scale copies (tcgen05.cp) itself, right before its MMAs (same-thread tcgen05 ops run
                          // in order): one mbarrier hop less per stage (the stage's hold time bounds the weight stream: Little's law)
#endif
#ifndef TPB2_GATHER
#define TPB2_GATHER 1   // routed FC1 B operand gathered by TMA (tile::gather4) from the token-major x codes: x quantization no longer waits
                        // for the routing tables (it runs next to the gate), no per-panel scatter
#endif
#ifndef TPB2_FC2STATIC
#define TPB2_FC2STATIC 0   // FC2 tickets in static panel order (ticket u -> panel u / nrg): the producer streams a unit's weights at once, only
                           // the B warp waits for the panel's h (its FC1 item count); rr / stream-K complete panels about in index order
#endif
#ifndef TPB2_FC2S_LO
#define TPB2_FC2S_LO 1   // (v62: M1..8 -- with SHW1 / SHGU / RRSPLIT 1 the static order wins at M1 -4.5%, M2 -2.4%, M8 -1%, M6 ~0 (k291/k292))   // FC2STATIC for TPB2_FC2S_LO <= M <= TPB2_FC2S_HI (v53): the W2 stages stream before the panel's h is ready.  Paired
#endif                   // (k225-k227): M4 -1.0/-3.8/-7.6/-4.2/-6.4/-5.5/-4.3% (mean -4.7%); M5 +1.3%, M6 / M8 +-1% (two runs, opposite signs),
#ifndef TPB2_FC2S_SHF
#define TPB2_FC2S_SHF 0   // (FC2STATIC) static order with the shared panels first (their h is ready first at small M, their FC2 is 2x the bytes):
                          // paired (k230, static at M1-8) worse everywhere: M4 +7..10% vs shared-last, M1 +3.5..7.6% -> off
#endif
#ifndef TPB2_FC2S_HI     // M1 +4..7%, M2 +1.5%, M12 +1..2% -> M4 only
#define TPB2_FC2S_HI 8
#endif
#ifndef TPB2_T1W
#define TPB2_T1W 0   // T1 by warp 7 (idle once the MMA warp issues the SF copies) per (token, 256-row tile) as soon as its FC2 contributions are
                     // counted (red.release per FC2 unit, no return): T1 overlaps the FC2 tail instead of waiting for every CTA's FC2
#endif
#ifndef TPB2_T1W_SLEEP
#define TPB2_T1W_SLEEP 300   // ns between T1 warp polling sweeps
#endif
#ifndef TPB2_T1W_MODE
#define TPB2_T1W_MODE 0   // (diagnostics) 1: warp 7 only counts (every pair to the tail); 2: counts without the gpu fence (WRONG ordering)
#endif
#ifndef TPB2_WARM
#define TPB2_WARM 0   // epilogue warps: one dry pass of the routed FC1 and the FC2 epilogue code right after the role split (no waits, no global
                      // side effects): the kernel is 286 KB of code, the first real pass otherwise runs i-cache cold (measured ~2x)
#endif
#ifndef TPB2_WARM_WAIT
#define TPB2_WARM_WAIT 0   // (WARM) the dry passes wait for the first work descriptor (the producer started); 2: only when that descriptor is
                           // FC1, after its second stage landed (the producer / MMA loops are running warm; the epilogue would idle in acc_wait):
                           // paired (k214) M1 +3.5/+4.2%, M2-8 ~0 -- the cold dry pass (3-4 us) outlasts the stream and delays the real epilogue
#endif
#ifndef TPB2_T1STATIC
#define TPB2_T1STATIC 1   // tail T1 items statically b, b + G, .. (no queue atomic)
#endif
#ifndef TPB2_L2ALL_MAXM
#define TPB2_L2ALL_MAXM 0   // M <= this: right after the tables every CTA requests its slice of ALL the layer's needed weights (routed FC1 + FC2 +
                            // scales, shared expert) into L2 in 64 KB bulk prefetches: one DRAM round for everything, the TMA loads then hit L2
#endif
#ifndef TPB2_ROLL
#define TPB2_ROLL 1   // FC1 epilogue SwiGLU / h quantization as compact loops (cold-code size on the critical path)
#endif
#ifndef TPB2_LLMERGE
#define TPB2_LLMERGE 0   // split-K partials as (fp32 bits, epoch) 64-bit pairs: the owner polls the data itself (one L2 round trip, no flag)
#endif
#ifndef TPB2_SHEARLY
#define TPB2_SHEARLY 1   // fused shared expert with nshq = 16 Psh quarter pieces <= G / 2: CTA j < nshq runs shared quarter j first in both FC1 modes
                         // (stream-K: the routed stages over CTAs nshq .. G - 1) and issues its first TPB2_SHPRE stages before the PDL wait (the
                         // weights depend on nothing): the shared FC1 off the routing chain.  The prologue scratch sits above those stages
#endif
#ifndef TPB2_SHPRE
#define TPB2_SHPRE 2
#endif
#ifndef TPB2_SHQ
#define TPB2_SHQ 4   // shared FC1 pieces per item (rr / SHEARLY): 4 quarters of 8 chunks, or 8 eighths of 4 (power of 2)
#endif
#ifndef TPB2_RRSPLIT
#define TPB2_RRSPLIT 1   // (v59: 1 -- CORRECTNESS: routed K-halves at M1 are not trtllm-gen's chain: 1/200 sweep offsets differed (116 elements); whole
                         // items 0; costs M1 ~+1.7%)  rr with all work in one wave (small M): every routed item in up to this many K-pieces on consecutive CTAs, merged by the
                         // last-arriving piece (LASTM counters)
#endif
#ifndef TPB2_PAIR
#define TPB2_PAIR 0   // CTA pairs (cluster 2, tcgen05 cta_group::2, M = 256 MMAs): the leader schedules 256-row work units for both and issues
                      // the MMAs; each CTA streams its own 128-row tile + half the token rows (SFB: all of them); the peer's warp 1 forwards its
                      // stage arrivals to the leader (FULLP), its epilogue releases the leader's accumulators (ACCE x 16) and ring slots (WE)
#endif
#ifndef TPB2_CSPLIT
#define TPB2_CSPLIT 0   // 2-CTA clusters (cta_group::1 MMAs): every routed FC1 item in two K-halves on the cluster's CTAs, rank 1's fp32 partial
                        // stored into rank 0's smem (DSMEM) + one remote arrive, rank 0 adds it (piece order) and finishes the item; FC1 entries
                        // round-robin over the clusters (shared pieces as before, consecutive CTAs + LASTM); FC2 per CTA as before
#endif
#ifndef TPB2_RTPUB
#define TPB2_RTPUB 1   // (SFCP_MMA, !T1W) warp 7 publishes each FC2 unit's 256-row-tile counts (fence + red) as the epilogue queues them, instead of
                       // the epilogue publishing all of a CTA's counts at its end: T1 starts sooner, no fence on the epilogue warps
#endif
#ifndef TPB2_RRTAIL2
#define TPB2_RRTAIL2 0   // rr: the entries of the last partial wave (routed items) split in K over S = PG / tail (<= 4) consecutive units,
                         // merged by the last-arriving piece (LASTM; partial slots MAXG + b).  Paired A/B: +2.5..8% M16-64 (end-of-FC1
                         // merges under full DRAM load are slow) -> off
#endif
#ifndef TPB2_FC2DIRECT
#define TPB2_FC2DIRECT 0   // (paired A/B: +13..21% M64-128 -- 2-B stores) FC2 epilogue: each lane stores its row's bf16 per token straight to outp / shout (a warp's 32 lanes = 32 consecutive
                           // rows of the token: 64 contiguous bytes) instead of the smem transpose + 2 named barriers per row tile
#endif
#ifndef TPB2_FC2PAIR
#define TPB2_FC2PAIR 1   // (LEANMMA 2) an FC2 row tile's two stages in one issue round: both waited, 16 MMAs, two EMPTY commits
#endif                   // (paired A/B: -0.2..-2.1% M8-128)
#ifndef TPB2_FC1PAIR
#define TPB2_FC1PAIR 0   // (LEANMMA 2) routed FC1 pieces' stages in pairs per issue round as well
#endif
#ifndef TPB2_LASTM
#define TPB2_LASTM 1   // shared quarter pieces (rr / SHEARLY stream-K): every quarter stores its fp32 partial and counts itself in (acq_rel atomic on a
                       // never-reset counter, mod 4); the LAST to arrive sums the four in quarter order (== owner TMEM partial + contributors in CTA
                       // order, bitwise) and finishes the item -- no CTA waits on flags (the owner used to block its epilogue for the slowest quarter)
#endif
#ifndef TPB2_MERGE_BS
#define TPB2_MERGE_BS 3   // shared FC1 split-K merge: contributors' partials loaded per round trip
#endif
#ifndef TPB2_LEANMMA
#define TPB2_LEANMMA 2   // MPAIR = 0: a lean one-stage MMA issue loop (2: hoisted operands, incremental stage / parity)
#endif
#ifndef TPB2_RRTAIL
#define TPB2_RRTAIL 0   // rr: whole waves round-robin; the last partial wave's items split in K over S = 4 | 2 | 1 consecutive CTAs (S * tail <= G)
                        // with split-K merges (partials in their own slots MAXG + b): no CTA does one item more than the others
#endif
#ifndef TPB2_TABW
#define TPB2_TABW 0   // (v64: 0 -- after the v57-v63 small-M changes the CTA-wide tables win at M1: paired -1.9%, 6/9, k296/k297)  small fused gate, M <= TPB2_TABW_MAXM: routing tables by one warp (no CTA-wide phases)
#endif
#ifndef TPB2_TABW_MAXM
#define TPB2_TABW_MAXM 1   // (v56 paired k241: M1 -2.9/-2.1%, M2 ~0, M4 +0.8..3.4%, M8 +7..10% -> M1 only)
#endif
#ifndef TPB2_DBG
#define TPB2_DBG 0   // phase stamps / timers compiled in (b2_kdbg builds with 1); 0: no dbg code in the kernel
#endif
#define DBGP (TPB2_DBG ? P.dbg_ : static_cast<unsigned long long*>(nullptr))
#ifndef TPB2_TRACE
#define TPB2_TRACE 0   // Xtrace-style phase trace (xtrace.cuh / xtrace.py; b2_layer_ab --trace): dbg_ = the xtrace buffer (TPB2_DBG must be 0)
#endif
#define XT_ON TPB2_TRACE
#ifndef XT_FMT
#define XT_FMT TPB2_TRACE   // (TPB2_TRACE=2: the XTrace 8-B bounded layout; TPB2_XT_MASK = XT_MASK site groups)
#endif
#if defined(TPB2_XT_MASK) && !defined(XT_MASK)
#define XT_MASK TPB2_XT_MASK
#endif
#include "xtrace.cuh"
#ifndef TPB2_PF1
#define TPB2_PF1 0   // small fused gate: warp 7 derives this CTA's first FC1 work from the gate ids (no tables) and L2-prefetches its first two
                     // stages while the tables are built (cold-L2 first-stage DRAM latency off the critical path; 32 KB per CTA)
#endif
#ifndef TPB2_LL_SLEEP
#define TPB2_LL_SLEEP 200
#endif
#ifndef TPB2_PROBE
#define TPB2_PROBE 0   // timing probes (WRONG results): 1 = no SF copies (tcgen05.cp), 2 = no MMAs, 4 = no B operand copies, 8 = no routed SFA bulk copies,
                       // 16 = SF copies only for the first 6 NS (stale finite scales later: no NaN slow paths)
                       // 128 = no FC1 SFB store + proxy fence, 256 = relaxed h-publish counters, 512 = no rt-publish fence, 1024 = no FC2 outp stores,
                       // 2048 = no T1 outp / shout loads
#endif
#ifndef TPB2_PGATE
#define TPB2_PGATE 1   // fused gate, M > SMALLM: a token gated by 8 warps together (per-warp top-6 + merge; bitwise == gate_token)
#endif
#ifndef TPB2_QW811
#define TPB2_QW811 0   // (PGATE) x quantization on warps 8..11 only (warps 0..7 gate meanwhile)
#endif
#ifndef TPB2_L2W2_MAXM
#define TPB2_L2W2_MAXM 0   // M <= this: after the tables, every panel's FC2 weights prefetched into L2 (sliced over the grid)
#endif
#ifndef TPB2_T1L_MAXM
#define TPB2_T1L_MAXM 0   // (v63: 0 -- after SHW1 / SHGU / static FC2 at M1 the generic tail wins: paired -2.8%, 7/8, k295)  M <= this (<= SMALLM): T1 by the last FC2 contributor -- warp 7 counts every queued FC2 unit's (token, 256-row
                          // tile) contributions with acq_rel atomics and T1s the pairs it completes at once (no row-tile publish / poll / claim)
#endif
#ifndef TPB2_HPUB
#define TPB2_HPUB 0   // FC1 h publishes (pan_cnt count + ready list) by the warp-7 publisher (one fence per batch): the epilogue goes on to
                      // the next item instead of waiting ~1.5 us on its acq_rel atomic
#endif
#ifndef TPB2_PGATE1
#define TPB2_PGATE1 0   // small fused gate, M = 1: the token by warps 0..7 together (gate_token_cta)
#endif
#ifndef TPB2_LMV
#define TPB2_LMV 0   // LASTM split-K partials row-major (token columns contiguous): float4 stores / loads -- the merge code (cold, on the
                     // M1-2 critical path) ~4x smaller
#endif
#ifndef TPB2_FC2G8
#define TPB2_FC2G8 0   // FC2 units of 8 row tiles when there are >= 2 G of them (half the unit boundaries; FC2VAR tail unchanged)
#endif
#ifndef TPB2_FC2GMAX
#define TPB2_FC2GMAX 8   // cap on the FC2 unit width (row tiles)
#endif
#ifndef TPB2_DLY_SITE
#define TPB2_DLY_SITE 0   // causal-profiling probe: spin TPB2_DLY_NS at one site (1 after the tables, all threads; 2 before the tail;
#endif                    // 3 each routed FC1 act (epilogue); 4 each FC2 row tile epilogue; 5 the shared FC1 items' epilogue)
#ifndef TPB2_DLY_NS
#define TPB2_DLY_NS 2000
#endif
__device__ __forceinline__ void dly_spin(unsigned ns) {
    unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); if (t - t0 >= ns) break; }
}
#ifndef TPB2_PFF
#define TPB2_PFF 0   // > 0: L2-prefetch this many first stages of the CTA's first routed FC1 piece once the panels are known (mid-tables)
#endif
#ifndef TPB2_RPRE
#define TPB2_RPRE 0   // > 0 (small fused gate): warp 7 derives this CTA's first routed FC1 piece from the gate ids (the tables' panel order:
                      // experts ascending) and issues its first TPB2_RPRE stages (weights + SFA) into ring stages 0.. before the tables;
                      // the producer skips them (first-stage DRAM latency off the FC1 start, ~0.8 us at M1-8)
#endif
#ifndef TPB2_PUBRED
#define TPB2_PUBRED 0   // static FC2 order (the B warp polls pan_cnt, the ready list is unused): the h publish is a red.release (no atomic
                        // round trip on the epilogue's publishing lane) instead of atom.acq_rel + ready-list append
#endif
#ifndef TPB2_RTMAJ
#define TPB2_RTMAJ 0   // > 0 (FC2VAR): the FC2 row tiles of the last TPB2_RTMAJ slots (>= 1000: the slots from the pairs region on) go in
                       // 256-row-tile-major order (row pairs of every late slot, then the next 256 rows): T1 tiles complete one by one at the
                       // end of FC2, so CTAs out of FC2 tickets T1 + all-reduce the finished tiles while the rest of FC2 runs
#endif
#ifndef TPB2_FC2CAP
#define TPB2_FC2CAP 0   // >0: a unit claims at most ceil(FC2 units / PG) + FC2CAP - 1 FC2 tickets (idle units no longer take every ticket while the
                        // FC1 units stream: at M1 92 idle CTAs took all 224 tiles, up to 4 each, the 56 FC1 CTAs none)
#endif
#ifndef TPB2_FC2RLX
#define TPB2_FC2RLX 0   // FC2 tickets claimed two ahead and the ready-list slots read relaxed (no acquire stall in the producer's issue path);
                        // the B warp acquires the panel's FC1 count itself before its h copies
#endif
#ifndef TPB2_OUTL2
#define TPB2_OUTL2 0   // FC2 outputs (outp / shout) stored with an L2 evict_last policy (the T1 reads them at the end, after ~100 MB of weights)
#endif
#ifndef TPB2_ACTX
#define TPB2_ACTX 0   // (ROLL) routed FC1 act: 4 tokens per loop round (independent chains interleaved) and the per-token warp amax by one
                      // redux.max on the bits of |a| (== the 5-step fmaxf butterfly for non-negative floats) instead of 5 dependent shuffles
#endif
#ifndef TPB2_SHW1
#define TPB2_SHW1 1   // (v57: 1 -- CORRECTNESS: shared quarters (M < 4) are NOT bitwise vs DeepGEMM's one K chain: --xoff 37 gave ~1100
                      // mismatches at M1/M2; whole items 0 over 24 inputs; costs M1 ~+10%, M2 ~+5%; SHL2 prefetch does not recover it)
                      // M >= this (v43: 4; paired vs 8 at M4 -1.5/-0.5%, M6 -7.0/-0.7%; M1 / M2 +6..11% with whole items): every shared FC1 item whole on one CTA (one K chain: no quarter partials / LASTM merge -- merges in the
                      // streaming phase take 10+ us of congested loads and block the epilogue; == DeepGEMM bit for bit at M 24-64, fixing the
                      // ~1100 shared mismatches at M >= 48).  Paired vs quarters: M8 -2.9%, M20-48 -3..-7%, M64-128 ~0 (+ SHW1_W: -2..-6%);
                      // M1 / M4 +7..9% (quarters keep their parallelism below)
#endif
#ifndef TPB2_SHW1_W
#define TPB2_SHW1_W 3   // (SHW1) waves of routed items the whole shared items' CTAs skip (a whole item ~ 3 routed items of time)
#endif
#ifndef TPB2_FC2VAR
#define TPB2_FC2VAR 1   // FC2 tickets over slot-major row tiles with a finer tail: units of fc2g (4) row tiles, then the last ~3 G row tiles
                        // as G units of 2 and G single row tiles (CTAs finish FC2 within ~1 row tile of each other instead of a 4-tile unit;
                        // paired A/B on two nodes: M24 -3.7%, M32 -4.3%, M64 -1.7..-2.7%, M96 -3.4%, M128 -1.1..-2.3%, M16 +0.8%; bitwise)
#endif
#ifndef TPB2_XRLX
#define TPB2_XRLX 0   // x flags before the tables: relaxed loads (all in flight together; one fence.acq_rel at the check) instead of ld.acquire each
#endif
#ifndef TPB2_GUNR
#define TPB2_GUNR 0   // gate: activation + routing-weight loops unrolled (ILP; the rolled loops are latency-serial)
#endif
#ifndef TPB2_T1LL_MAXM
#define TPB2_T1LL_MAXM 0   // M <= this (fused or no shared expert): the FC2 epilogue stores outp / shout as LL messages (6 bf16 + epoch tag per 16 B,
                           // 22 per 128-row tile) in the tcnt tail, and warp 7 of CTA j T1s (token, 256-row tile) j by polling its 7 x 44 messages:
                           // no counts, no fences, no queue (T1L's acq_rel count after the last FC2 tile took 1.3-1.8 us, its loads another 1 us).
                           // Bitwise, but paired M1 +0.6 us (k212): the poll sees the last message 1.7-3 us after the store, the cold sum 0.7-1.2 us,
                           // sends 0.95 us -> off (T1L stays)
#endif
#ifndef TPB2_PROBE_T1LOCAL
#define TPB2_PROBE_T1LOCAL 0   // PROBE (wrong output): the T1's reduce-scatter sends go to this rank's own buffer, no all-reduce (NVLink cost of T1)
#endif
#ifndef TPB2_MC
#define TPB2_MC 0   // NVLS multicast pushes (multimem.st, one store reaches the 4 ranks; receivers still sum in rank order): 1 the one-shot
                    // T1 sends, 2 the all-gather pushes, 4 the one-shot T1 sends + a local store of the own copy (host: rs / ag pointer lists
                    // carry the multicast address as entry NDEV; b2_layer_ab --mcbuf).  Bitwise but no gain (v77 runs, node -261): MC 1 M1
                    // +1.0 us (the slowest rank's own copy now waits for the switch), MC 4 M8 -0.1 / M16 0; b2_ar_bench: AG multicast slower at
                    // M64+ (the switch's own copy costs ingress, and ingress is the bound) -> off
#endif
#ifndef TPB2_ARV
#define TPB2_ARV 0   // all-reduce variants (bits, same arithmetic -> bitwise): 1 T3 polls of a thread's messages in flight together; 2 T2 + T3 in
                     // one poll loop (a thread's T2 sources and T3 messages in flight together, T3 never waits behind the CTA's T2); 4 no fence
                     // before the end-of-kernel epoch count; 8 explicit in-flight source polls (issue all pending loads, then test -- the
                     // pending-mask loop compiled to 4 dependent L2 round trips per poll); 16 one-shot: each pair received by the warp that
                     // T1'd it (no local copy, no CTA-wide receive); 32 epoch from a u64 CTA-exit count (red, no fence / returning atomic at
                     // exit).  Paired (KS vs KS2, us): 8 M8 -0.37 / M32 -0.43 / M128 -0.4; 4 M8 -0.53 / M128 -0.67; 32 vs 4 M8 -0.5 / M128
                     // -0.6; 2 M64 -0.5 / M128 -0.9; 1 M128 -0.35; 16 M1 0 / M8 0 / M4 +0.55 (off).  v77: 40 one-shot builds, 34 RS + AG
#endif
#ifndef TPB2_IDPRE
#define TPB2_IDPRE 0   // build_tables (M > SMALLM): the global ids loaded into registers before the clear phase (one L2 round trip hidden)
#endif
#ifndef TPB2_LLS
#define TPB2_LLS 0   // (suite_b200) all-reduce messages = 8 bf16 per 16 B, arrival = w != 0xFFFFFFFF, slots re-armed by their reader
#endif
#ifndef TPB2_TABDBG
#define TPB2_TABDBG 0   // (suite_b200, trace only) per-warp points in the build_tables bits phase
#endif
#ifndef TPB2_ENTT
#define TPB2_ENTT 0   // (suite_b200) build_tables entries per token (one thread: all slots of its token)
#endif
#ifndef TPB2_IDV
#define TPB2_IDV 0   // (suite_b200) build_tables, ids given: int4 vector loads of all ids, every load in flight before the clear phase
#endif
#ifndef TPB2_OUTPAD
#define TPB2_OUTPAD 0   // outp row stride DIM + this (elements): breaks the 8 KB power-of-two stride of the T1's (token, slot) rows (64: paired ~0)
#endif
#ifndef TPB2_EPIFIRST
#define TPB2_EPIFIRST 0   // the producer starts this CTA's FC2 loads only after the epilogue processed all its FC1 descriptors (the last FC1 item's
                          // act + h publish otherwise run under the CTA's own FC2 stream: M32 trace 6.4 us act + 1.7 us publish on the last panel).
                          // Paired (k221): +1.5..+5.3% at M8-128 -- the FC2 overlap is worth more than the faster last act -> off
#endif
#ifndef TPB2_T1L_MINM
#define TPB2_T1L_MINM 0   // > 0: T1L also for TPB2_T1L_MINM <= M <= 128 (pairs that complete early are T1'd during FC2, the rest at the end).
                          // Bitwise but 2x SLOWER at M48-128 (k222): warp 7's per-unit acq_rel counts (~1.5 us per 32) + serial T1 back up the queue
#endif
#ifndef TPB2_EARLYTRIG
#define TPB2_EARLYTRIG 0   // griddepcontrol.launch_dependents at the kernel start: the next PDL kernel launches now and its CTAs take the SMs our
                           // CTAs free (its own griddepcontrol.wait still waits for this grid's completion and memory flush)
#endif
#ifndef TPB2_PROBE_EMPTY
#define TPB2_PROBE_EMPTY 0
#endif
#ifndef TPB2_LLRR
#define TPB2_LLRR 1   // M <= TPB2_LLRR_MAXM: rr K-pieces merged by piece 0 (static owner) polling the others' epoch-tagged 64-bit partials
#endif                // (no store + fence + atomic + reload round trips of LASTM; same summation order).  v55 paired (k235/k236): M1 -2.0%
#ifndef TPB2_LLRR_MAXM   // (14 samples, 11 favourable), bitwise; M2 +2.5% (shared quarters: the owner polls 3) -> M1 only
#define TPB2_LLRR_MAXM 1
#endif
#ifndef TPB2_FC2L
#define TPB2_FC2L 0   // M = 1 (fused shared, one-shot AR): CTA r < 32 runs FC2 row tile r of every panel (W2 streamed before h is ready, the B
                      // warp waits for each panel's h), keeps token 0's outputs in smem and T1s its 128 rows locally (no counts, no reloads);
                      // the one-shot messages are numbered per 128-row half (22 each; message 43 of token 0 uses token 1's slot 0).
                      // Bitwise but +14% at M1 (k237/t238): 7 serial units per CTA each wait ~1 us for their h B-load (FC2 19.6 -> 28.7 us),
                      // and the cold local T1 + sends took 4.7 us -> off
#endif
#ifndef TPB2_SHL2
#define TPB2_SHL2 0   // M <= TPB2_SHL2_MAXM: L2 prefetch of the shared gate_up weights at the kernel start (whole shared items stream from L2)
#endif
#ifndef TPB2_SHL2_MAXM
#define TPB2_SHL2_MAXM 3
#endif
#ifndef TPB2_SHGU
#define TPB2_SHGU 1   // M <= TPB2_SHGU_MAXM (whole shared items): a shared item's gate chain and up chain on two CTAs (units 2 i, 2 i + 1), each
                      // one full K chain (== DeepGEMM, bitwise); a ring stage = two K chunks of the CTA's half (full 32 KB slots: the per-SM
                      // stream is in-flight bound); the up CTA hands its fp32 tile to the gate CTA as epoch-tagged 64-bit words.
                      // v58 paired (k254): M1 -3.8/-5.0/-2.8%, M3 -2.5%, M2 -0.8/+0.3/+1.9%; bitwise at xoff 0 / 37 -> M1
#endif
#ifndef TPB2_SHGU_LO2
#define TPB2_SHGU_LO2 0   // (a second M range for SHGU)
#endif
#ifndef TPB2_SHGU_HI2
#define TPB2_SHGU_HI2 -1
#endif
#ifndef TPB2_SHGU_MAXM
#define TPB2_SHGU_MAXM 6   // (v60: M2 paired -2.6% over 11 samples; v61: M4 -2.2% / M6 -1.7% over 9 samples each (k285/k286), M8 ~0)
#endif
#ifndef TPB2_SILU
#define TPB2_SILU 1   // routed SwiGLU as u * __fdividef(g, 1 + __expf(-g)) (== trtllm-gen: reciprocal-based).  v59: the IEEE-division form (0) gave
#endif                // the rare "110-element" mismatches at every M (M128: 40/40 sweep offsets, M2 2/300); variants 1/2/5/6 -> 0 everywhere
#ifndef TPB2_PROBE_END
#define TPB2_PROBE_END 0   // timing probe (WRONG output): 1 = stop after FC1 (no FC2, T1, all-reduce), 2 = stop after FC2 (no T1 / all-reduce)
#endif
#ifndef TPB2_H1U
#define TPB2_H1U 0   // FC2: a unit's h (all NSTG2 stages' B chunks + SFB) loaded ONCE into one of two buffers in the B / SFB ring regions (free once
                     // this CTA's FC1 stages are consumed) and used by all its row tiles (was: reloaded per row tile, per stage).
                     // Bitwise, no hang, but paired k281 mean +0.8% (M8 -2.0/+4.4, M64 +2.8/+1.0, M128 +1.6/-0.7) -> off
#endif
#ifndef TPB2_FC2G4_MINM
#define TPB2_FC2G4_MINM 64   // FC2 unit width capped at 4 row tiles for M >= this (paired k282/k283: M64-128 -0.1..-1.7%, M48 +1%)
#endif
#ifndef TPB2_SHEARLY_OFFM
#define TPB2_SHEARLY_OFFM 2   // (v65) no pre-PDL shared stages for M <= this: paired M1 -1.3%, M2 -1.6% (8 samples each, k300-k302)
#endif
#ifndef TPB2_FC2PAIR_OFFM
#define TPB2_FC2PAIR_OFFM 8   // (v65) FC2 row tiles stage by stage for M <= this: paired M1-6 ~-1%, M8 -2.3% (k300-k302); M16 +1.5%
#endif
#ifndef TPB2_ACTX_MINM
#define TPB2_ACTX_MINM 0   // (v66: 0 -- both act paths in one build cost M1-4 ~1.5-2.5% (k307: code size); use a separate -DTPB2_ACTX=1 build for M >= 32 instead)  routed FC1 act in 4-token rounds (ACTX) for M >= this: paired M32 -3.9%, M48 -0.7%, M64 -1.1%, M96 -1.8%,
#endif                      // M128 -1.25% (k304/k305); M4 / M16 / M20 slower (narrow panels)
#ifndef TPB2_T1LL_SLEEP
#define TPB2_T1LL_SLEEP 128
#endif
#ifndef TPB2_ONESHOT_MAXM
#define TPB2_ONESHOT_MAXM 48   // one-shot LL all-reduce for M <= this, else reduce-scatter + all-gather
#endif
#ifndef TPB2_MMAX
#define TPB2_MMAX 0   // > 0: this build serves M <= TPB2_MMAX only (the host checks): device reads of M clamped to it, so the large-M
                      // paths (multi-CTA routing tables, RS + AG all-reduce, ...) fold away at compile time (less cold i-cache code)
#endif
#ifndef TPB2_SPEC
#define TPB2_SPEC 0   // 1: the production configuration fixed at compile time (TP4, top-6, fused gate + fused shared expert, no in-kernel dsv3
                      // router; the host checks): its runtime branches fold away like TPB2_MMAX's.  (suite_b200) 2: routed-only (TP4,
                      // top-6, ids / weights given: no gate, no shared expert)
#endif
#ifndef TPB2_WHOLE
#define TPB2_WHOLE 0   // 1: every FC1 work entry is a whole item (rr always, RRSPLIT 1, no RRTAIL2, SHW1 1: s0 == 0, s1 == all stages), so the
                       // epilogue's split-K paths (non-owner partial stores, owner merges) are compiled out
#endif
#ifndef TPB2_T1CNT
#define TPB2_T1CNT 0   // 1: the T1 publisher counts a batch's FC2 row tiles per 256-row tile as byte fields of two u64 (SWAR) instead of
                       // the 16 x 8 unrolled compares (~380 SASS on the FC2 -> T1 path)
#endif
#ifndef TPB2_QX1
#define TPB2_QX1 0   // 1 (+ TPB2_MMAX, TPB2_GRID): quant_x with one unit per warp when 8 TPB2_MMAX <= its warps (not 4 units in flight: smaller code)
#endif
#ifndef TPB2_PUT1
#define TPB2_PUT1 0   // 1: the producer's routed FC1 entries (SHW1 waves, then round robin) from one put site (one inlined copy of issue)
#endif
#ifndef TPB2_ACTU
#define TPB2_ACTU 0   // 1 (+ TPB2_MMAX <= 16): FC1 epilogue SwiGLU / h quantization unrolled over the <= TPB2_MMAX live tokens (no ROLL rotations)
#endif
#define TPB2_ACTU_ON (TPB2_ACTU && TPB2_MMAX && TPB2_MMAX <= 16)
#ifndef TPB2_GRID
#define TPB2_GRID 0   // > 0: the grid (CTAs) fixed at compile time (the host checks; divisions by G fold to multiplies)
#endif
#ifndef TPB2_SFBI
#define TPB2_SFBI 0   // 1: routed FC1 x scales (SFB) staged per ITEM (two item buffers [16 stages][32 rows][16 B] after the tables) in
                      // batches of 4 stages with ONE proxy fence per batch, instead of a generic store + proxy fence per stage on the B
                      // warp (probe 128: those cost ~7% of the FC1 phase at M128).  The MMA warp copies stage s of item buffer (item & 1).
#endif
#ifndef TPB2_SFBI_SB
#define TPB2_SFBI_SB 4   // (SFBI) stages per batch (4 x 16-B rows in registers per lane, x2 with the next batch in flight)
#endif
#ifndef TPB2_SHPOL
#define TPB2_SHPOL 0   // L2 policy of the SHARED expert weight TMA loads (re-streamed per 32-token panel at M > 32): 0 evict_first (as the
                       // routed weights), 1 evict_normal, 2 evict_last -- the panels' concurrent re-reads of a tile can then hit L2
#endif
#ifndef TPB2_RRB
#define TPB2_RRB 0   // 1: in-kernel router for M > 64 in 64-row blocks (units = 16 expert groups x K-parts x row blocks; each row's K16
                     // chain unchanged -> logits still == cuBLAS's split-K bitwise; x per unit halved at M128)
#endif
#ifndef TPB2_RANK
#define TPB2_RANK -1   // >= 0: this build serves TP rank TPB2_RANK only (the host checks): peer indexing at compile time
#endif
#ifndef TPB2_SPEC_RF
#define TPB2_SPEC_RF 0   // 1: TPB2_SPEC keeps the in-kernel dsv3 router (rF, M <= 4) as a runtime option
#endif
#ifndef TPB2_RS
#define TPB2_RS -1   // in-kernel split-K router: -1 runtime (rS), 0 compiled out (the host checks)
#endif
#ifndef TPB2_MMIN
#define TPB2_MMIN 1   // > 1: this build serves M >= TPB2_MMIN only (the small-M paths fold away likewise)
#endif

namespace TPB2_NS {

#ifndef TPB2_NDEV
#define TPB2_NDEV 4   // (suite_b200) TP size: 4 (v77) or 8 (DSV4-Flash TP8: INTER 256)
#endif
constexpr int NDEV = TPB2_NDEV;
static_assert(NDEV == 4 || NDEV == 8, "TP4 / TP8");
constexpr int DIM = 4096, INTER = 2048 / NDEV, NEXP = 256, TOPK_MAX = 8, MAXM = 512;
constexpr int MAXE = MAXM * 6;               // entries (token, expert) upper bound at top-6 (asserted on the host)
#ifndef TPB2_NMAX
#define TPB2_NMAX 32
#endif
constexpr int NMAX = TPB2_NMAX;              // token panel width = MMA N bound (8: a small-M build -- 2 KB B stages, room for more ring stages)
constexpr int MAXP = 512;                    // panels upper bound (<= NEXP + MAXE / NMAX)
constexpr int NT1 = 2 * INTER / 128;         // 8 FC1 tiles (128 w13 rows = 64 gate + 64 up) per panel
constexpr int NCH1 = DIM / 128;              // 32 FC1 k-chunks
constexpr int NRT2 = DIM / 128;              // 32 FC2 row tiles
constexpr int NCH2 = INTER / 128;            // 4 FC2 k-chunks
constexpr int KC = 2;                        // k-chunks per pipeline stage
constexpr int NSTG1 = NCH1 / KC, NSTG2 = NCH2 / KC;   // 16, 2 stages per FC1 item / FC2 unit
#ifndef TPB2_NS_STAGES
#define TPB2_NS_STAGES 4
#endif
constexpr int NS = TPB2_NS_STAGES;           // ring stages (5 x 16 KB of weights in flight per SM)
constexpr int THREADS = 384;                 // 12 warps: producer, MMA, 2 x 4 epilogue, B, SF
// FC2 unit = fc2g consecutive 128-row tiles of one panel (one ticket / ring entry); fc2g (1 | 2 | 4) chosen per launch from the panel count
constexpr int NRT = DIM / 256;               // 16 T1 / all-reduce row tiles (256 rows)
constexpr int LRT = NRT / NDEV;              // 4 row tiles per reduce-scatter owner
constexpr int NMSG = 43;                     // LL messages per (row tile, token): 6 bf16 + tag each, the last carries 4
constexpr int MAXG = 160;
#ifndef TPB2_ACC
#define TPB2_ACC 4
#endif
constexpr int ACC = TPB2_ACC, ACC_COLS = 64;   // TMEM accumulator buffers (routed: columns 0..31; shared FC1: gate 0..31, up 32..63)
constexpr int MAXPS = MAXM / NMAX;           // shared-expert panels (tokens 32 sp .. 32 sp + 31)
constexpr int NT1S = INTER / 128;            // 4 shared FC1 items per panel (128 h columns: gate rows 128 k.., up rows INTER + 128 k..)
constexpr int NSTG1S = NCH1;                 // 32 shared FC1 stages per item (one 128-k chunk of both tiles)
// scale factors per ring stage: SFA chunks at columns 0..7 (4 per chunk), SFB at TSF_B .. TSF_B + 3 (16 columns per stage)
constexpr int TMEM_COLS = 512, TSF_STRIDE = 16, TSF_B = 8, TSF0 = TMEM_COLS - NS * TSF_STRIDE;
static_assert(ACC * ACC_COLS <= TSF0, "TMEM: accumulators + scale factors");
constexpr int WR = 16;                       // work ring depth
// shared memory (1024-aligned base): A [NS][KC][16 KB] | B [NS][KC][NMAX][128] | SFA [NS][KC][512] | SFB [NS][512] | tables | ring | bars
constexpr int A_BYTES = KC * 16384, B_BYTES = KC * NMAX * 128;
constexpr int OFF_A = 0, OFF_B = OFF_A + NS * A_BYTES, OFF_SFA = OFF_B + NS * B_BYTES, OFF_SFB = OFF_SFA + NS * KC * 512;
constexpr int OFF_TAB = OFF_SFB + NS * 512;
constexpr int TAB_ENT = 0;                             // uint16 [MAXE] entry: token | top-k slot mask << 9 (top-k <= 6)
constexpr int TAB_PINF = TAB_ENT + MAXE * 2;           // uint32 [MAXP] panel: e | n << 8 | ebase << 14
constexpr int TAB_RW = TAB_PINF + MAXP * 4;            // float [TPB2_SMALLM * 8] small fused gate: routing weights of (t, k)
constexpr int TAB_SIDS = TAB_RW + TPB2_SMALLM * 8 * 4; // int16 [TPB2_SMALLM * 8] small fused gate: expert ids of (t, k) (T1 reads them here)
constexpr int TAB_AM = TAB_SIDS + TPB2_SMALLM * 8 * 2;   // float [4][NMAX] FC1 epilogue: per-warp amax exchange
constexpr int TAB_MISC = TAB_AM + 4 * NMAX * 4;        // int [32]: 0 U, 1 entries, 2 panels, 8 T1 item, 9 epoch
constexpr int TAB_PRDY = TAB_MISC + 32 * 4;            // uint32 [MAXP / 32] (unused since v15)
constexpr int TAB_LANES = TAB_PRDY + MAXP / 8;          // uint8 [128] FC2 row of TMEM lane
constexpr int TAB_STG = TAB_LANES + 128;               // bf16 [2][16][128] FC2 epilogue transpose staging
constexpr int TAB_T1S = TAB_STG + 2 * 16 * 128 * 2;   // bf16 [256] T1 warp staging
constexpr int T1Q = 64;                                // FC2-unit completion queue (epilogue -> T1 warp)
constexpr int TAB_T1Q = TAB_T1S + ((TPB2_T1W || TPB2_T1L_MAXM) ? 256 * 2 : 0);             // int [T1Q] unit descriptors (d.x << 24 | d.y), [T1Q] = count, [T1Q + 1] = epilogue done,
                                                       // [T1Q + 2] = consumed
constexpr int TAB_RING = TAB_T1Q + (T1Q + 4) * 4;      // int4 [WR] work ring
constexpr int TAB_BAR = TAB_RING + WR * 16;            // uint64 barriers
constexpr int NBAR = 4 * NS + 2 * ACC + 2 * WR + 4 + 6 + 4;   // (+ 4 H1U: h buffer 0 / 1 full, empty)   // (+ 6 in-kernel router: W, x buffer 0, MMA done, x buffer 1, buffer 0 / 1 free)   // (+ NS PAIR FULLP: the peer's stage arrivals, forwarded; + 4 CSPLIT MFULL / MEMPTY)
constexpr int TAB_MBUF = (TAB_BAR + NBAR * 8 + 15) & ~15;   // float [2][NMAX][128] CSPLIT: the cluster peer's FC1 partials (rank 0)
constexpr int TAB_BYTES = TAB_MBUF + (TPB2_CSPLIT ? 2 * NMAX * 128 * 4 : 0);
constexpr int OFF_SFBI = (OFF_TAB + TAB_BYTES + 1023) & ~1023;   // (TPB2_SFBI) uint8 [2][NSTG1][512] routed FC1 SFB images per item
constexpr int SFBI_BYTES = TPB2_SFBI ? 2 * NSTG1 * 512 : 0;
constexpr int SMEM_TOTAL = (TPB2_SFBI ? OFF_SFBI + SFBI_BYTES : OFF_TAB + TAB_BYTES), SMEM_ALLOC = SMEM_TOTAL + 1024;
static_assert(SMEM_ALLOC <= 232448, "smem budget");
static_assert(!TPB2_SFBI || (TPB2_LEANMMA == 2 && !TPB2_MPAIR && !TPB2_FC1PAIR && !TPB2_PAIR && TPB2_GATHER && TPB2_SFCP_MMA), "SFBI: the lean 1-CTA gather path");
static_assert(!TPB2_SFBI || TPB2_WHOLE, "SFBI: item images are written before the piece's first EMPTY wait -- needs whole FC1 items (>= NS stages per piece; split / stream-K pieces would race the MMA still reading piece rfi - 2)");
static_assert(!(TPB2_DBG && TPB2_TRACE), "the dbg_ buffer is either the stamps or the xtrace buffer");
static_assert(!(TPB2_LASTM && TPB2_T1W), "TPB2_LASTM counts shared quarters in P.tcnt (the T1W counters)");
static_assert(!TPB2_RRTAIL && !TPB2_PF && !TPB2_PF2 && !TPB2_PF1, "removed in v36");
static_assert(!(TPB2_CSPLIT && TPB2_PAIR), "CSPLIT or PAIR");
static_assert(!TPB2_PAIR || (TPB2_LEANMMA == 2 && TPB2_SFCP_MMA && !TPB2_MPAIR && !TPB2_T1W && TPB2_GATHER && !TPB2_LLMERGE && !TPB2_WARM),
              "PAIR: the LEANMMA 2 MMA loop, SF copies by the MMA warp, gather B, flag merges only");
static_assert(OFF_B % 1024 == 0 && B_BYTES % 1024 == 0 && OFF_TAB % 16 == 0 && TAB_BAR % 8 == 0, "alignment");
// prologue scratch inside the A ring (dead before the first TMA)
constexpr int PRO_BITS = TPB2_SHEARLY ? TPB2_SHPRE * A_BYTES : 0;   // uint32 [16 words][NEXP] token bits per expert (above the early stages)
constexpr int PRO_PRE = PRO_BITS + 16 * NEXP * 4;      // uint16 [16][NEXP] exclusive word prefix
constexpr int PRO_OFFE = PRO_PRE + 16 * NEXP * 2;      // int [NEXP] entry offset of expert
constexpr int PRO_PIDE = PRO_OFFE + NEXP * 4;          // int [NEXP] first panel of expert
constexpr int PRO_WS = PRO_PIDE + NEXP * 4;            // int [32] warp sums
constexpr int PRO_GIDS = PRO_WS + 32 * 4;              // int [TPB2_SMALLM * 8] small fused gate: ids
constexpr int PRO_IDS = PRO_GIDS + TPB2_SMALLM * 8 * 4; // uint16 [MAXM][8] expert of (t, k), 0xFFFF = none (tables only)
constexpr int PRO_TKE = PRO_IDS + MAXM * 8 * 2;         // uint16 [MAXM][8] (t, k) -> panel << 5 | row (x scatter only)
static_assert(PRO_TKE + MAXM * 8 * 2 <= NS * A_BYTES && PRO_IDS % 16 == 0, "prologue scratch");

struct Params {
    CUtensorMap map1, map2;                  // w13 [E * 1024 rows][2048 B], w2 [E * 4096 rows][256 B]: 16U4_ALIGN16B {128, rows, chunks}
    const uint8_t* w13_sf;                   // [E][1024][128] block_scale_interleave bytes: 512 B per (tile, chunk)
    const uint8_t* w2_sf;                    // [E][4096][16]
    const uint8_t* wp[4];                    // raw w13 [E][1024][2048], w2 [E][4096][256], shared gate_up [1024][4096], down [4096][512] (L2 prefetch)
    const uint8_t* lanes;                    // [128] FC2 output row of TMEM lane (per 128-row tile)
    const __nv_bfloat16* xb;                 // [M][4096] MoE input
    const float* glog;                       // fused gate: router logits fp32 [M][256] (nparts > 1: [nparts][M][256] K-part sums, added in order), else null
    int nparts, pstride;
    const float* gbias;                      // [256] score correction bias
    int* ids;                                // [M][topk] (fused gate: written here)
    float* rw;                               // [M][topk]
    int topk;
    const __nv_bfloat16* shared;             // [M][4096] shared expert output (alt stream) or null
    float factor;                            // routed scaling factor (1.5)
    uint8_t* xq;                             // [MAXM][4096] e4m3
    uint8_t* xs;                             // [MAXM][128] ue8m0
    uint8_t* xg;                             // [MAXP][NCH1][NMAX][128] x per panel row, K-major SW128 (FC1 B operand images)
    uint8_t* xgsf;                           // [MAXP][NSTG1][512] their SFB images (lane = row, byte 8 (c & 1) + kb)
    unsigned long long* dbg_;                // [grid][32] globaltimer phase stamps (0..15) + FC2 wait / busy sums in ns (16..23), or null
    uint8_t* himg;                           // [MAXP][NCH2][NMAX][128] h, K-major SW128
    uint8_t* hsf;                            // [MAXP][NSTG2][512] h SFB images (lane = token, byte 8 (c & 1) + kb)
    float* part;                             // [MAXG][NMAX][128] FC1 stream-K partials
    unsigned* part_flags;                    // [MAXG] epoch tags
    unsigned* xflag;                         // [2][MAXM] quantized 512-element groups of token t (8 = done; epoch parity)
    unsigned* gcnt;                          // [2] gated tokens (epoch parity)
    unsigned* pan_cnt;                       // [2][MAXP] FC1 items done per panel
    unsigned* rdy;                           // [2][MAXP + 32] panels in the order their h became ready (p + 1; 0 = not yet), [2][MAXP] = slots taken
    unsigned* fc2_next;                      // [4] FC2 unit queue head [par], T1 item queue head [2 + par]
    unsigned* rt_cnt;                        // [2][NRT] FC2 units stored per 256-row tile
    unsigned* work;                          // [0] epoch, [1 + parity] CTA done counters
    __nv_bfloat16* outp;                     // [MAXM][TOPK_MAX][4096] per-slot expert outputs
    uint4* rs[NDEV];                         // peer reduce-scatter buffers (two parity halves)
    uint4* ag[NDEV];                         // peer all-gather buffers (two parity halves)
    uint4* rsmc;                             // (TPB2_MC) NVLS multicast address of the rs buffers (a store reaches all ranks), or null
    uint4* agmc;                             // (TPB2_MC) multicast address of the ag buffers, or null
    __nv_bfloat16* out;                      // [M][4096]
    int M, my_rank, ndev, oneshot;
    // fused shared expert (shw): the production FP8 DeepseekV2MLP slice in place (DeepGEMM numerics: per-128 UE8M0 scales)
    int shw;
    CUtensorMap map1s, map2s;                // gate_up [1024][4096] e4m3: box {128 B, 128 rows, 1 chunk}; down [4096][512]: box {128, 128, 2}
    CUtensorMap mapxq, mapxsf;               // TPB2_GATHER: xq [MAXM][4096] box {128 B, 1} SW128; xsf2 [MAXM][NSTG1 x 16 B] box {16 B, 1} (gather4 rows)
    unsigned* tcnt;                          // [2][MAXM][NRT] FC2 contributions counted per (token, 256-row tile) (TPB2_T1W)
    uint8_t* xsf2;                           // [MAXM][NSTG1][16] x scales by token in the SFB row layout (byte 8 (c & 1) + kb of stage c / 2)
    const uint8_t* sfa1s;                    // [8 row blocks][32 k blocks][512] SFA images (one UE8M0 byte per 128 x 128 block, replicated)
    const uint8_t* sfa2s;                    // [32][4][512]
    uint8_t* xgs;                            // [MAXPS][NCH1][NMAX][128] x per (token, 128-group) quantized (ceil rule, eps 1e-10), SW128
    uint8_t* xgsfs;                          // [MAXPS][NSTG1][512] its SFB images (4 equal bytes per 128-group)
    uint8_t* hsimg;                          // [MAXPS][NCH2][NMAX][128] shared h (silu_and_mul_clamp bf16 -> per-128 FP8)
    uint8_t* hssf;                           // [MAXPS][NSTG2][512]
    __nv_bfloat16* shout;                    // [MAXM][4096] shared expert output (bf16), added in T1
    // in-kernel router (rS > 0; 5 <= M <= 32): units (16-expert group, K-part) on CTAs G - 1 - u; K-part chains == cuBLAS split-K bitwise
    CUtensorMap maprx, maprw;                // x bf16 {64, M, 64 chunks} box {64, R, rch}; router W bf16 {64, 256, 64} box {64, 16, rch} (SW128)
    float* rpart;                            // [rS][M][256] fp32 K-part logits (= glog; the gate sums them in order)
    int rS, rR, rch, rk0[4], rkn[4];         // parts, x rows R (M rounded to 8), box chunks (longest part)
    int rB;                                  // (TPB2_RRB) 64-row blocks (1 otherwise)
    int rxch;                                // x chunks per TMA load (< rch: double-buffered x sub-parts, M 33..64)
    // in-kernel dsv3 router (rF; M <= 4): production's sglang dsv3_router_gemm (FMA chains) bit for bit, logits -> rpart [M][256]
    const __nv_bfloat16* rwf;                // router W bf16 [256][4096]
    int rF;
};
__device__ __forceinline__ int pM(const Params& P) { return max(TPB2_MMAX ? min(P.M, TPB2_MMAX) : P.M, TPB2_MMIN); }   // (TPB2_MMAX: range known to the compiler)
__device__ __forceinline__ int pOS(const Params& P) { return (TPB2_MMAX && TPB2_MMAX <= TPB2_ONESHOT_MAXM) ? 1 : (TPB2_MMIN > TPB2_ONESHOT_MAXM ? 0 : P.oneshot); }
__device__ __forceinline__ int pND(const Params& P) { return TPB2_SPEC ? NDEV : P.ndev; }   // (TPB2_SPEC / TPB2_RS: compile-time values)
__device__ __forceinline__ int pTK(const Params& P) { return TPB2_SPEC ? 6 : P.topk; }
__device__ __forceinline__ int pSHW(const Params& P) { return TPB2_SPEC == 2 ? 0 : TPB2_SPEC ? 1 : P.shw; }
__device__ __forceinline__ const __nv_bfloat16* pSHD(const Params& P) { return TPB2_SPEC ? nullptr : P.shared; }
__device__ __forceinline__ bool pGL(const Params& P) { return TPB2_SPEC == 2 ? false : TPB2_SPEC ? true : P.glog != nullptr; }
__device__ __forceinline__ int pRF(const Params& P) { return (TPB2_SPEC && !TPB2_SPEC_RF) ? 0 : P.rF; }
__device__ __forceinline__ int pRS(const Params& P) { return TPB2_RS == 0 ? 0 : P.rS; }
__device__ __forceinline__ int pRK(const Params& P) { return TPB2_RANK >= 0 ? TPB2_RANK : P.my_rank; }
__device__ __forceinline__ int pNP(const Params& P) { return (TPB2_SPEC && TPB2_RS == 0) ? 1 : P.nparts; }   // (production router: one logits part)

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_wait_cl(uint32_t b, uint32_t parity) {   // (acquire at cluster scope: data / arrivals from the peer CTA)
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar, uint64_t hint = 0x12F0000000000000ull) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(hint) : "memory");
}
constexpr uint64_t SH_HINT = TPB2_SHPOL == 2 ? 0x14F0000000000000ull : (TPB2_SHPOL == 1 ? 0x1000000000000000ull : 0x12F0000000000000ull);
// PAIR: the peer's weight tile, its completion counted on the leader's barrier (cluster address)
__device__ __forceinline__ void tma3d_cg2(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t cluster_bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(cluster_bar), "l"(0x12F0000000000000ull) : "memory");
}
__device__ __forceinline__ void pf_tma3d(const CUtensorMap* map, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(map), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void pf_bulk(const void* p, uint32_t bytes) { asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory"); }
__device__ __forceinline__ void tma_gather4(uint32_t dst, const CUtensorMap* map, int col, int r0, int r1, int r2, int r3, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                 ::"r"(dst), "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_v4_hint(void* p, uint4 v) {   // (TPB2_OUTL2: L2 evict_last)
    if (TPB2_OUTL2) {
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
        asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol) : "memory");
    } else {
        *reinterpret_cast<uint4*>(p) = v;
    }
}
__device__ __forceinline__ uint4 ld_v4_last_use(const void* p) {   // (TPB2_OUTL2: the T1's read demotes the line again: L2 evict_first)
    if (TPB2_OUTL2) {
        uint4 v;
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
        asm volatile("ld.global.cg.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol) : "memory");
        return v;
    }
    return __ldcg(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ void st_relaxed64(unsigned long long* p, unsigned long long v) { asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ unsigned long long ld_relaxed64(const unsigned long long* p) {
    unsigned long long v; asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
#ifndef TPB2_ACQPOLL
#define TPB2_ACQPOLL 1   // flag polls by ld.acquire (one L2 round trip once set; 0: relaxed polls + an acquiring re-load, two)
#endif
__device__ __forceinline__ void wait_ge(const unsigned* p, unsigned v) {
    if (TPB2_ACQPOLL) { while (ld_acquire(p) < v) {} } else { while (ld_relaxed(p) < v) {} (void)ld_acquire(p); }
}
__device__ __forceinline__ void wait_eq(const unsigned* p, unsigned v) {
    if (TPB2_ACQPOLL) { while (ld_acquire(p) != v) {} } else { while (ld_relaxed(p) != v) {} (void)ld_acquire(p); }
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_relaxed_add(unsigned* p, unsigned v) { asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_release_add(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) {
    unsigned r; asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r;
}
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void st_ll(uint4* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void st_mc(uint4* p, uint4 v) {   // NVLS multicast store: the switch writes the 16 B to every rank's buffer
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(__uint_as_float(v.x)), "f"(__uint_as_float(v.y)),
                 "f"(__uint_as_float(v.z)), "f"(__uint_as_float(v.w)) : "memory");
}
__device__ __forceinline__ uint4 ld_ll(const uint4* p) { uint4 v; asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ bool elect_one() {
    uint32_t pred = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned long long ctime() { return static_cast<unsigned long long>(clock64()); }   // (in-loop sums: cycles, cheap)
__device__ __forceinline__ uint32_t warp_uniform(uint32_t v) { return __shfl_sync(0xffffffffu, v, 0); }
// routed FC1 activation: up * silu(gate) on the clamped values (TPB2_SILU: candidate formulations of trtllm-gen's, for the bitwise sweep)
__device__ __forceinline__ float silu_mul(float g, float u) {
#if TPB2_SILU == 1
    return u * __fdividef(g, 1.0f + __expf(-g));
#elif TPB2_SILU == 2
    return u * (g * __frcp_rn(1.0f + __expf(-g)));
#elif TPB2_SILU == 3
    return u * (g / (1.0f + expf(-g)));
#elif TPB2_SILU == 4
    return (u * g) / (1.0f + __expf(-g));
#elif TPB2_SILU == 5
    return u * __fdividef(g, 1.0f + expf(-g));
#elif TPB2_SILU == 6
    return u * (g * __fdividef(1.0f, 1.0f + __expf(-g)));
#elif TPB2_SILU == 7
    return __fdividef(u * g, 1.0f + __expf(-g));
#else
    return u * (g / (1.0f + __expf(-g)));
#endif
}
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) { __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&r); }
// MXFP8 scale exponent: 2^ceil(log2(amax / 448)) as an unbiased exponent (amax 0 -> the smallest normal scale)
__device__ __forceinline__ int ue8m0_exp(float amax) {
    const uint32_t b = __float_as_uint(fmaxf(amax * (1.0f / 448.0f), 1.17549435e-38f));
    return static_cast<int>((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
}
// h (FC1 output): the OCP MX rule 2^(floor(log2 amax) - 8) (e4m3 emax 8; values above 448 saturate) -- trtllm-gen's, measured
__device__ __forceinline__ int ue8m0_exp_floor(float amax) {
    return max(-127, static_cast<int>((__float_as_uint(amax) >> 23) & 0xFF) - 127 - 8);
}
__device__ __forceinline__ float exp2i(int e) { return __uint_as_float(static_cast<uint32_t>(e + 127) << 23); }
__device__ __forceinline__ uint32_t e4m3_of(float v) { return static_cast<uint32_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)); }

// tcgen05
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
    return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
           (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (static_cast<uint64_t>(layout) << 61);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) { return smem_desc(addr, 0, 1024, 2); }
__device__ __forceinline__ uint64_t desc_sf(uint32_t addr) { return smem_desc(addr, 0, 128, 0); }
// kind::mxf8f6f4: A e2m1 (5), B e4m3 (0), fp32 D, UE8M0 scales, K-major, M = 128; N >> 3 at bit 17; scale byte ids
__device__ __forceinline__ uint32_t idesc_f4f8(int n, uint32_t sf_id, uint32_t atype = 5u) {   // atype 5 e2m1 (routed), 0 e4m3 (shared)
    return (sf_id << 4) | (atype << 7) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) | (((TPB2_PAIR ? 256u : 128u) >> 4) << 24) | (sf_id << 29);
}
#if TPB2_PAIR
#define TPB2_CG "cta_group::2"
#else
#define TPB2_CG "cta_group::1"
#endif
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
    if (TPB2_PROBE & 2) return;
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
                 " tcgen05.mma." TPB2_CG ".kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
                 ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, uint64_t desc) { if (TPB2_PROBE & 1) return; asm volatile("tcgen05.cp." TPB2_CG ".32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(desc) : "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t bar) {   // (PAIR: multicast to the same barrier in both CTAs)
#if TPB2_PAIR
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"(static_cast<uint16_t>(3)) : "memory");
#else
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
#endif
}
// cluster (PAIR) helpers: the peer CTA's shared address of a local one, remote mbarrier arrive, remote 16-B store
__device__ __forceinline__ uint32_t peer_addr(uint32_t a, uint32_t prank) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(prank)); return r;
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_bar) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote_relaxed(uint32_t cluster_bar) {   // (no release: async-proxy data already complete)
    asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_bar) : "memory");
}
__device__ __forceinline__ void st_cluster_f32(uint32_t cluster_addr, float v) {
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(cluster_addr), "f"(v) : "memory");
}
__device__ __forceinline__ void st_cluster_v4(uint32_t cluster_addr, int4 v) {
    asm volatile("st.shared::cluster.v4.s32 [%0], {%1, %2, %3, %4};" ::"r"(cluster_addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_ld8(uint32_t taddr, float* v) {
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// 16 columns of the warp's TMEM lane quadrant AND the wait in one asm block: tcgen05.ld's destination registers are undefined until
// tcgen05.wait::ld, so the compiler must not see them before it (separate asm statements let it copy them early -- read stale values)
__device__ __forceinline__ void tc_ld16_wait(uint32_t taddr, float* v) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
                 "tcgen05.wait::ld.sync.aligned;"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr) : "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}

// ------------------------------------------------------------------------------------------------ fused gate (tp4moe2, bitwise sglang moe_fused_gate)
__device__ __forceinline__ float tri_act(float x) {
    float r;
    asm("{\n\t"
        ".reg .pred ps, pf, pz, pg;\n\t"
        ".reg .b32 a, e, y, y2, ys, eb, t, mb, fe, f, p, q, lg;\n\t"
        "mul.f32 a, %1, 0f3FB8AA3B;\n\t"
        "ex2.approx.f32 e, a;\n\t"
        "add.f32 y, e, 0f3F800000;\n\t"
        "setp.lt.f32 ps, y, 0f00800000;\n\t"
        "mul.f32 y2, y, 0f4B000000;\n\t"
        "selp.f32 ys, y2, y, ps;\n\t"
        "selp.f32 eb, 0fC1B80000, 0f00000000, ps;\n\t"
        "add.s32 t, ys, -1059760811;\n\t"
        "and.b32 t, t, -8388608;\n\t"
        "sub.s32 mb, ys, t;\n\t"
        "cvt.rn.f32.s32 fe, t;\n\t"
        "fma.rn.ftz.f32 fe, fe, 0f34000000, eb;\n\t"
        "add.f32 f, mb, 0fBF800000;\n\t"
        "fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3E0F2955;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n\t"
        "mul.f32 q, f, p;\n\t"
        "fma.rn.ftz.f32 q, q, f, f;\n\t"
        "fma.rn.ftz.f32 lg, fe, 0f3F317218, q;\n\t"
        "setp.lt.u32 pf, ys, 2139095040;\n\t"
        "@!pf fma.rn.ftz.f32 lg, ys, 0f7F800000, 0f7F800000;\n\t"
        "setp.eq.f32 pz, ys, 0f00000000;\n\t"
        "selp.f32 lg, 0fFF800000, lg, pz;\n\t"
        "setp.gt.f32 pg, %1, 0f41A00000;\n\t"
        "selp.f32 lg, %1, lg, pg;\n\t"
        "sqrt.approx.ftz.f32 %0, lg;\n\t"
        "}" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ uint32_t gate_key(float v) { const uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
__device__ __forceinline__ void gate_cas(uint64_t& a, uint64_t& b) { const uint64_t x = a > b ? a : b, y = a > b ? b : a; a = x; b = y; }
constexpr int GATE_TOPK = 6;
// one token by one warp: lanes < topk return (expert id, routing weight) of slot lane
__device__ __forceinline__ void gate_token(const Params& P, const float* lg, int np, int lane, int& id, float& wt, Xt* xt = nullptr) {
#if TPB2_ROLL   // the same arithmetic as the #else branch as compact loops over rotating register arrays (cold code on the prologue's critical path)
    float lv[8], bv[8], sc[8];
    uint64_t h[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) { lv[w] = __ldcg(lg + lane + 32 * w); bv[w] = __ldg(P.gbias + lane + 32 * w); sc[w] = 0.f; h[w] = 0ull; }
    if (np > 1) {   // (our router: cuBLAS's K-parts summed in order, fp32; every part's loads in flight together, then the ordered adds)
        float lp[3][8];
#pragma unroll
        for (int p = 1; p < 4; ++p)
#pragma unroll
            for (int w = 0; w < 8; ++w) lp[p - 1][w] = p < np ? __ldcg(lg + static_cast<size_t>(p) * P.pstride + lane + 32 * w) : 0.f;
#pragma unroll
        for (int p = 1; p < 4; ++p)
#pragma unroll
            for (int w = 0; w < 8; ++w) if (p < np) lv[w] = lv[w] + lp[p - 1][w];
    }
#if TPB2_GUNR   // (8 independent activation chains interleaved: the rolled loop serializes ~150-cycle dependent chains)
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const float a = tri_act(lv[w]);
        float v = __fadd_rn(a, bv[w]);
        v = v == v ? v : -1e30f;
        sc[w] = a;
        h[w] = (static_cast<uint64_t>(gate_key(v)) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(lane + 32 * w));
    }
#else
#pragma unroll 1
    for (int w = 0; w < 8; ++w) {   // (input at [0], result appended at [7]: after 8 rounds element w sits at [w])
        const float a = tri_act(lv[0]);
        float v = __fadd_rn(a, bv[0]);
        v = v == v ? v : -1e30f;
        const uint64_t hw = (static_cast<uint64_t>(gate_key(v)) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(lane + 32 * w));
#pragma unroll
        for (int k = 0; k < 7; ++k) { lv[k] = lv[k + 1]; bv[k] = bv[k + 1]; sc[k] = sc[k + 1]; h[k] = h[k + 1]; }
        sc[7] = a; h[7] = hw;
    }
#endif
    if (xt && lane == 0) xt->rec(XT_PT(61));
    gate_cas(h[0], h[2]); gate_cas(h[1], h[3]); gate_cas(h[4], h[6]); gate_cas(h[5], h[7]);
    gate_cas(h[0], h[4]); gate_cas(h[1], h[5]); gate_cas(h[2], h[6]); gate_cas(h[3], h[7]);
    gate_cas(h[0], h[1]); gate_cas(h[2], h[3]); gate_cas(h[4], h[5]); gate_cas(h[6], h[7]);
    gate_cas(h[2], h[4]); gate_cas(h[3], h[5]);
    gate_cas(h[1], h[4]); gate_cas(h[3], h[6]);
    gate_cas(h[1], h[2]); gate_cas(h[3], h[4]); gate_cas(h[5], h[6]);
    if (xt && lane == 0) xt->rec(XT_PT(62));
    int ex[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) ex[k] = 0;
#pragma unroll 1
    for (int k = 0; k < GATE_TOPK; ++k) {
        const uint32_t hk = static_cast<uint32_t>(h[0] >> 32), hl = static_cast<uint32_t>(h[0]);
        const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
        const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
        if (hk == gm && hl == gl) {
#pragma unroll
            for (int i = 0; i < 7; ++i) h[i] = h[i + 1];
            h[7] = 0ull;
        }
#pragma unroll
        for (int i = 0; i < GATE_TOPK - 1; ++i) ex[i] = ex[i + 1];
        ex[GATE_TOPK - 1] = static_cast<int>(0xFFFFFFFFu - gl);
    }
    if (xt && lane == 0) xt->rec(XT_PT(63));
    float R[GATE_TOPK];
#if TPB2_GUNR   // (6 independent lookups)
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const int ew = ex[k] >> 5;
        float sv = sc[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) sv = ew == w ? sc[w] : sv;
        R[k] = __shfl_sync(0xffffffffu, sv, ex[k] & 31);
    }
    if (false)
#else
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) R[k] = 0.f;
#endif
#pragma unroll 1
    for (int k = 0; k < GATE_TOPK; ++k) {   // (ex rotates through once: back in order afterwards)
        const int e = ex[0];
#pragma unroll
        for (int i = 0; i < GATE_TOPK - 1; ++i) { ex[i] = ex[i + 1]; R[i] = R[i + 1]; }
        ex[GATE_TOPK - 1] = e;
        const int ew = e >> 5;
        float sv = sc[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) sv = ew == w ? sc[w] : sv;
        R[GATE_TOPK - 1] = __shfl_sync(0xffffffffu, sv, e & 31);
    }
#else
    float sc[8];
    uint64_t h[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const float a = tri_act(__ldcg(lg + lane + 32 * w));
        float v = __fadd_rn(a, __ldg(P.gbias + lane + 32 * w));
        v = v == v ? v : -1e30f;
        sc[w] = a;
        h[w] = (static_cast<uint64_t>(gate_key(v)) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(lane + 32 * w));
    }
    gate_cas(h[0], h[2]); gate_cas(h[1], h[3]); gate_cas(h[4], h[6]); gate_cas(h[5], h[7]);
    gate_cas(h[0], h[4]); gate_cas(h[1], h[5]); gate_cas(h[2], h[6]); gate_cas(h[3], h[7]);
    gate_cas(h[0], h[1]); gate_cas(h[2], h[3]); gate_cas(h[4], h[5]); gate_cas(h[6], h[7]);
    gate_cas(h[2], h[4]); gate_cas(h[3], h[5]);
    gate_cas(h[1], h[4]); gate_cas(h[3], h[6]);
    gate_cas(h[1], h[2]); gate_cas(h[3], h[4]); gate_cas(h[5], h[6]);
    int ex[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const uint32_t hk = static_cast<uint32_t>(h[0] >> 32), hl = static_cast<uint32_t>(h[0]);
        const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
        const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
        ex[k] = static_cast<int>(0xFFFFFFFFu - gl);
        if (hk == gm && hl == gl) {
#pragma unroll
            for (int i = 0; i < 7; ++i) h[i] = h[i + 1];
            h[7] = 0ull;
        }
    }
    float R[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const int ew = ex[k] >> 5;
        float sv = sc[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) sv = ew == w ? sc[w] : sv;
        R[k] = __shfl_sync(0xffffffffu, sv, ex[k] & 31);
    }
#endif
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(R[0], R[1]), R[2]), R[3]), __fadd_rn(R[4], R[5]));
    const float norm = sum > 0.0f ? sum : 1.0f;
    float r = R[0];
    id = ex[0];
#pragma unroll
    for (int k = 1; k < GATE_TOPK; ++k) { r = lane == k ? R[k] : r; id = lane == k ? ex[k] : id; }
    wt = div_full(r, norm);
}
#ifndef TPB2_GWARM
#define TPB2_GWARM 0   // one shared (noinline) copy of the gate code, run once on dummy logits (the bias) by warp 3 before griddep_wait:
                       // the prologue's gate then runs i-cache warm (diagnostic: cold-code vs logits-latency share of the gate)
#endif
struct GateR { int id; float wt; };
__device__ __noinline__ GateR gate_call(const Params& P, const float* lg, int np, int lane, Xt* xt) {
    GateR r;
    gate_token(P, lg, np, lane, r.id, r.wt, xt);
    return r;
}
// TPB2_PGATE: one token gated by warps 0..7 together (expert 32 warp + lane): per-warp top-6 by redux, then warp 0 merges the 48
// candidates -- the same keys (score bits, then the lower expert index), selection order and weight arithmetic as gate_token
__device__ __forceinline__ void gate_token_cta(const Params& P, const float* lg, int np, int warp, int lane, uint64_t* cand, float* sca,
                                               int& id, float& wt) {
    const int e = 32 * warp + lane;
    float lv = __ldcg(lg + e);
    const float bv = __ldg(P.gbias + e);
    if (np > 1) {
        float lp[3];
#pragma unroll
        for (int p = 1; p < 4; ++p) lp[p - 1] = p < np ? __ldcg(lg + static_cast<size_t>(p) * P.pstride + e) : 0.f;
#pragma unroll
        for (int p = 1; p < 4; ++p) if (p < np) lv = lv + lp[p - 1];
    }
    const float a = tri_act(lv);
    float v = __fadd_rn(a, bv);
    v = v == v ? v : -1e30f;
    uint64_t h = (static_cast<uint64_t>(gate_key(v)) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(e));
    sca[e] = a;
#pragma unroll 1
    for (int k = 0; k < GATE_TOPK; ++k) {
        const uint32_t hk = static_cast<uint32_t>(h >> 32), hl = static_cast<uint32_t>(h);
        const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
        const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
        if (lane == 0) cand[warp * GATE_TOPK + k] = (static_cast<uint64_t>(gm) << 32) | gl;
        if (hk == gm && hl == gl) h = 0ull;
    }
    named_bar(6, 256);
    if (warp != 0) return;
    uint64_t c0 = cand[lane], c1 = lane + 32 < 8 * GATE_TOPK ? cand[lane + 32] : 0ull;
    if (c1 > c0) { const uint64_t t = c0; c0 = c1; c1 = t; }
    int ex[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const uint32_t hk = static_cast<uint32_t>(c0 >> 32), hl = static_cast<uint32_t>(c0);
        const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
        const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
        ex[k] = static_cast<int>(0xFFFFFFFFu - gl);
        if (hk == gm && hl == gl) { c0 = c1; c1 = 0ull; }
    }
    float R[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) R[k] = sca[ex[k]];
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(R[0], R[1]), R[2]), R[3]), __fadd_rn(R[4], R[5]));
    const float norm = sum > 0.0f ? sum : 1.0f;
    float r = R[0];
    id = ex[0];
#pragma unroll
    for (int k = 1; k < GATE_TOPK; ++k) { r = lane == k ? R[k] : r; id = lane == k ? ex[k] : id; }
    wt = div_full(r, norm);
}

// MXFP8 quantization of x in units (token t, 512-element group cg) by one warp, lane l: elements 16 l .. + 15 (chunk 4 cg + l / 8, 16-B
// granule l % 8, 32-block = lanes l, l ^ 1).  Writes the reference copies xq / xs and the token's row of each of its panels' FC1 B images
// (SW128 granule + SFB word), then counts the unit on xcnt[t] (8 per token).  Units u = gw + i nw (token u / 8), 4 per warp in flight.
__device__ __forceinline__ void quant_x(const Params& P, const uint16_t* tke, unsigned* xcnt, int lane, int gw, int nw, bool scatter) {
    const int U = pM(P) * 8;
    const int QN = (TPB2_QX1 && TPB2_MMAX && 8 * TPB2_MMAX <= nw) ? 1 : 4;   // (units in flight; 1: every warp has <= 1 unit)
    for (int u0 = gw; u0 < U; u0 += QN * nw) {
        uint4 raw[4][2];
#pragma unroll
        for (int j = 0; j < QN; ++j) {
            const int u = u0 + j * nw;
            if (u < U) {
                const uint4* src = reinterpret_cast<const uint4*>(P.xb + static_cast<size_t>(u >> 3) * DIM + (u & 7) * 512) + lane * 2;
                raw[j][0] = __ldg(src); raw[j][1] = __ldg(src + 1);
            }
        }
#pragma unroll
        for (int j = 0; j < QN; ++j) {
            const int u = u0 + j * nw;
            if (u >= U) break;
            const int t = u >> 3, cg = u & 7, ch = 4 * cg + (lane >> 3), gr = lane & 7;
            const uint32_t w8[8] = {raw[j][0].x, raw[j][0].y, raw[j][0].z, raw[j][0].w, raw[j][1].x, raw[j][1].y, raw[j][1].z, raw[j][1].w};
            float am = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) am = fmaxf(am, fmaxf(fabsf(bf16lo(w8[i])), fabsf(bf16hi(w8[i]))));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
            const int ex = ue8m0_exp(am);
            const float inv = exp2i(-ex);
            uint32_t cw[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                cw[i] = e4m3_of(bf16lo(w8[2 * i]) * inv) | (e4m3_of(bf16hi(w8[2 * i]) * inv) << 8) | (e4m3_of(bf16lo(w8[2 * i + 1]) * inv) << 16) |
                        (e4m3_of(bf16hi(w8[2 * i + 1]) * inv) << 24);
            const uint4 code = make_uint4(cw[0], cw[1], cw[2], cw[3]);
            const uint32_t eb = static_cast<uint32_t>(ex + 127);
            uint32_t sfw = 0;   // the chunk's 4 scale bytes (blocks of lanes 8 c + 0, 2, 4, 6)
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) sfw |= (__shfl_sync(0xffffffffu, eb, (lane & ~7) + 2 * kb) & 0xFFu) << (8 * kb);
            if (pSHW(P)) {   // shared expert input: per (token, 128-group) e4m3, scale 2^ceil(log2(max(amax, 1e-10) / 448)) (sglang per_token_group_quant, ue8m0)
                float a128 = am;
                a128 = fmaxf(a128, __shfl_xor_sync(0xffffffffu, a128, 2));
                a128 = fmaxf(a128, __shfl_xor_sync(0xffffffffu, a128, 4));
                const int e128 = ue8m0_exp(fmaxf(a128, 1e-10f));
                const float i128 = exp2i(-e128);
                uint32_t sw[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    sw[i] = e4m3_of(bf16lo(w8[2 * i]) * i128) | (e4m3_of(bf16hi(w8[2 * i]) * i128) << 8) | (e4m3_of(bf16lo(w8[2 * i + 1]) * i128) << 16) |
                            (e4m3_of(bf16hi(w8[2 * i + 1]) * i128) << 24);
                const int sp = t >> 5, r = t & 31;
                uint8_t* row = P.xgs + ((static_cast<size_t>(sp) * NCH1 + ch) * NMAX) * 128 + (r >> 3) * 1024 + (r & 7) * 128;
                *reinterpret_cast<uint4*>(row + ((gr ^ (r & 7)) << 4)) = make_uint4(sw[0], sw[1], sw[2], sw[3]);
                if (gr == 0) *reinterpret_cast<uint32_t*>(P.xgsfs + (static_cast<size_t>(sp) * NSTG1 + (ch >> 1)) * 512 + r * 16 + 8 * (ch & 1)) =
                    static_cast<uint32_t>(e128 + 127) * 0x01010101u;
            }
            *reinterpret_cast<uint4*>(P.xq + static_cast<size_t>(t) * DIM + cg * 512 + lane * 16) = code;
            if (gr == 0) *reinterpret_cast<uint32_t*>(P.xs + static_cast<size_t>(t) * (DIM / 32) + ch * 4) = sfw;
            if (gr == 0 && !scatter) *reinterpret_cast<uint32_t*>(P.xsf2 + (static_cast<size_t>(t) * NSTG1 + (ch >> 1)) * 16 + 8 * (ch & 1)) = sfw;
#pragma unroll 1
            for (int k = 0; k < (scatter ? 8 : 0); ++k) {
                const unsigned v = tke[t * 8 + k];
                if (v == 0xFFFFu) continue;
                const int p = static_cast<int>(v >> 5), r = static_cast<int>(v & 31u);
                uint8_t* row = P.xg + ((static_cast<size_t>(p) * NCH1 + ch) * NMAX) * 128 + (r >> 3) * 1024 + (r & 7) * 128;
                *reinterpret_cast<uint4*>(row + ((gr ^ (r & 7)) << 4)) = code;
                if (gr == 0) *reinterpret_cast<uint32_t*>(P.xgsf + (static_cast<size_t>(p) * NSTG1 + (ch >> 1)) * 512 + r * 16 + 8 * (ch & 1)) = sfw;
            }
        }
    }
    // one fence for all of this warp's units, then their counts (fence + relaxed RMW = release)
    __syncwarp();
    if (lane == 0 && gw < U) {
        fence_acq_rel();
        for (int u = gw; u < U; u += nw) red_relaxed_add(xcnt + (u >> 3), 1u);
    }
}

// ------------------------------------------------------------------------------------------------ routing tables
struct Tabs {
    uint16_t* ids; uint16_t* ent; uint8_t* km; uint32_t* pinf; float* srw; float* am; int* misc; uint8_t* lanes; int4* ring;
    uint16_t* tke; uint32_t* prdy; __nv_bfloat16* stg; int16_t* sids;
};
__device__ __forceinline__ uint32_t xd0(const int4& d) { return (static_cast<uint32_t>(d.x & 0xF) << 28) | static_cast<uint32_t>(d.y & 0xFFFFFFF); }
__device__ __forceinline__ uint32_t xd1(const int4& d) { return (static_cast<uint32_t>(d.z & 0xFFFF) << 16) | static_cast<uint32_t>(d.w & 0xFFFF); }
__device__ __forceinline__ Tabs tabs(uint8_t* smem) {
    uint8_t* t = smem + OFF_TAB;
    return Tabs{reinterpret_cast<uint16_t*>(smem + PRO_IDS), reinterpret_cast<uint16_t*>(t + TAB_ENT), nullptr,
                reinterpret_cast<uint32_t*>(t + TAB_PINF), reinterpret_cast<float*>(t + TAB_RW), reinterpret_cast<float*>(t + TAB_AM),
                reinterpret_cast<int*>(t + TAB_MISC), t + TAB_LANES, reinterpret_cast<int4*>(t + TAB_RING),
                reinterpret_cast<uint16_t*>(smem + PRO_TKE), reinterpret_cast<uint32_t*>(t + TAB_PRDY), reinterpret_cast<__nv_bfloat16*>(t + TAB_STG),
                reinterpret_cast<int16_t*>(t + TAB_SIDS)};
}
// Every CTA, all THREADS: union = present experts in ascending id; entries of an expert = its tokens in ascending order (a token listing
// the same expert in several slots is ONE entry, km = its slots); panels of <= NMAX entries.  ids from smem (gids, small fused gate) or
// global (P.ids).  Writes T.ids, T.ent, T.km, T.pinf, misc 0..2.
struct NoAfter { __device__ void operator()() const {} };
// (TPB2_HPUB) hand a finished FC1 item's h publish (panel gp's count + the ready list) to the warp-7 publisher's queue
__device__ __forceinline__ void hpub_enqueue(uint8_t* smem, int gp, bool shared_item) {
    volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
    const int qn = q[T1Q];
    while (qn - q[T1Q + 2] >= T1Q) {}   // (queue full)
    q[qn % T1Q] = static_cast<int>(0x80000000u | (shared_item ? 0x40000000u : 0u) | static_cast<unsigned>(gp));
    __threadfence_block();
    q[T1Q] = qn + 1;
}
constexpr int IDV_MAXV = ((TPB2_MMAX ? TPB2_MMAX : MAXM) * 6 / 4 + THREADS - 1) / THREADS;   // (IDV) int4 id loads per thread
struct IdPre { int4 v[IDV_MAXV]; int on; };
__device__ __forceinline__ void idv_load(const Params& P, IdPre& q, int tid) {   // (IDV 2) this thread's int4 id loads, all in flight
    const int MT = pM(P) * pTK(P);
    q.on = pTK(P) == 6 && (MT & 3) == 0 && (reinterpret_cast<uintptr_t>(P.ids) & 15) == 0;
    if (!q.on) return;
#pragma unroll
    for (int j = 0; j < IDV_MAXV; ++j) {
        const int v = tid + j * THREADS;
        q.v[j] = v < MT / 4 ? __ldcg(reinterpret_cast<const int4*>(P.ids) + v) : make_int4(-1, -1, -1, -1);
    }
}
template <class After = NoAfter>
__device__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T, const int* gids, Xt* xt = nullptr, After after = After(), const IdPre* ipre = nullptr) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int M = pM(P), topk = pTK(P), MT = M * topk, W = (M + 31) >> 5;
    uint32_t* bits = reinterpret_cast<uint32_t*>(smem + PRO_BITS);
    uint16_t* pre = reinterpret_cast<uint16_t*>(smem + PRO_PRE);
    int* offe = reinterpret_cast<int*>(smem + PRO_OFFE);
    int* pide = reinterpret_cast<int*>(smem + PRO_PIDE);
    int* ws = reinterpret_cast<int*>(smem + PRO_WS);
    // (TPB2_IDPRE: global ids -- other CTAs' gate results -- loaded before the clear phase: their L2 round trip under it)
    int ide[2] = {-1, -1};
    const bool idpre = TPB2_IDPRE && !gids && MT <= 2 * THREADS;
    constexpr int MAXV = ((TPB2_MMAX ? TPB2_MMAX : MAXM) * 6 / 4 + THREADS - 1) / THREADS;   // (IDV) int4 loads per thread
    const bool idv = TPB2_IDV && !gids && topk == 6 && (MT & 3) == 0 && (reinterpret_cast<uintptr_t>(P.ids) & 15) == 0;
    int4 iv[MAXV];
    if (idv && ipre && ipre->on) {   // (IDV 2: loaded before the x quantization)
#pragma unroll
        for (int j = 0; j < MAXV; ++j) iv[j] = ipre->v[j];
    } else if (idv) {
#pragma unroll
        const int* ids_ = P.ids + (TPB2_TABDBG >= 2 ? static_cast<int>(blockIdx.x % (TPB2_TABDBG >= 2 ? TPB2_TABDBG : 1)) * MT : 0);   // (TABDBG >= 2: replicas)
        for (int j = 0; j < MAXV; ++j) {
            const int v = tid + j * THREADS;
            iv[j] = v < MT / 4 ? __ldcg(reinterpret_cast<const int4*>(ids_) + v) : make_int4(-1, -1, -1, -1);
        }
    }
    if (idpre) {
#pragma unroll
        for (int j = 0; j < 2; ++j) if (tid + j * THREADS < MT) ide[j] = __ldcg(P.ids + tid + j * THREADS);
    }
    for (int i = tid; i < W * NEXP; i += THREADS) bits[i] = 0u;
    for (int i = tid; i < M * 8; i += THREADS) { T.ids[i] = 0xFFFF; T.tke[i] = 0xFFFF; }
    for (int i = tid; i < MAXP / 32; i += THREADS) T.prdy[i] = 0u;
    __syncthreads();
    if (xt && threadIdx.x == 0) xt->rec(XT_PT(64));
    if (TPB2_TABDBG && idv) {   // (TABDBG) single-load latency probes
        if (xt && lane == 0) xt->rec(XT_PT(81));
        const int4 q1 = __ldcg(reinterpret_cast<const int4*>(P.ids) + lane + 32 * (blockIdx.x & 3));
        int s1 = __reduce_add_sync(0xffffffffu, q1.x ^ q1.w);
        if (xt && lane == 0) xt->rec(XT_PT(82), static_cast<uint32_t>(s1) & 0xFFu);
        const int4 q2 = __ldcg(reinterpret_cast<const int4*>(P.xb) + blockIdx.x * 32 + lane);
        s1 = __reduce_add_sync(0xffffffffu, q2.x ^ q2.w);
        if (xt && lane == 0) xt->rec(XT_PT(83), static_cast<uint32_t>(s1) & 0xFFu);
    }
    if (TPB2_TABDBG && idv) {   // (TABDBG) this warp's id loads landed
        int s_ = 0;
#pragma unroll
        for (int j = 0; j < MAXV; ++j) s_ += iv[j].x + iv[j].y + iv[j].z + iv[j].w;
        s_ = __reduce_add_sync(0xffffffffu, s_);
        if (xt && lane == 0) xt->rec(XT_PT(79), static_cast<uint32_t>(s_) & 0xFFu);
    }
    if (idv) {   // (IDV) the vector-loaded ids: entries p = 4 v .. 4 v + 3
#pragma unroll
        for (int j = 0; j < MAXV; ++j) {
            const int v = tid + j * THREADS;
            if (v >= MT / 4) continue;
            const int e4[4] = {iv[j].x, iv[j].y, iv[j].z, iv[j].w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int p = 4 * v + q, e = e4[q], t = p / 6, k = p - t * 6;
                if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
                    T.ids[t * 8 + k] = static_cast<uint16_t>(e);
                    atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
                }
            }
        }
    } else
    for (int p = tid; p < MT; p += THREADS) {
        const int e = idpre ? ide[(p - tid) / THREADS] : (gids ? gids[p] : __ldcg(P.ids + p));
        const int t = p / topk, k = p - t * topk;
        if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
            T.ids[t * 8 + k] = static_cast<uint16_t>(e);
            atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
        }
    }
    if (TPB2_TABDBG) { __syncwarp(); if (xt && lane == 0) xt->rec(XT_PT(80)); }   // (TABDBG) this warp's atomics issued
    __syncthreads();
    if (xt && threadIdx.x == 0) xt->rec(XT_PT(65));
    // per expert (tid == e, warps 0..7): count, word prefix; block scans of (present, count, panels)
    int c = 0, pres = 0, npan = 0, a = 0, b = 0, d = 0;
    if (tid < NEXP) {
        const int e = tid;
        for (int w = 0; w < W; ++w) { pre[w * NEXP + e] = static_cast<uint16_t>(c); c += __popc(bits[w * NEXP + e]); }
        pres = c > 0; npan = (c + NMAX - 1) / NMAX;
        a = pres; b = c; d = npan;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int ya = __shfl_up_sync(0xffffffffu, a, o), yb = __shfl_up_sync(0xffffffffu, b, o), yd = __shfl_up_sync(0xffffffffu, d, o);
            if (lane >= o) { a += ya; b += yb; d += yd; }
        }
        if (lane == 31) { ws[warp] = a; ws[8 + warp] = b; ws[16 + warp] = d; }
    }
    __syncthreads();
    if (xt && threadIdx.x == 0) xt->rec(XT_PT(66));
    if (tid < NEXP) {
        int xb = b - c, xd = d - npan;
#pragma unroll
        for (int w = 0; w < 8; ++w) if (w < warp) { xb += ws[8 + w]; xd += ws[16 + w]; }
        offe[tid] = xb; pide[tid] = xd;
        if (pres) {
            for (int p = 0; p < npan; ++p)
                T.pinf[xd + p] = static_cast<uint32_t>(tid) | (static_cast<uint32_t>(min(NMAX, c - p * NMAX)) << 8) | (static_cast<uint32_t>(xb + p * NMAX) << 14);
        }
    }
    if (tid == NEXP - 1) {
        int U = 0, E = 0, Pn = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) { U += ws[w]; E += ws[8 + w]; Pn += ws[16 + w]; }
        T.misc[0] = U; T.misc[1] = E; T.misc[2] = Pn;
    }
    __syncthreads();
    if (xt && threadIdx.x == 0) xt->rec(XT_PT(67));
    after();   // (panels known: T.pinf, T.misc[0..2])
    // entries: one per (token, expert) at its first slot
    if (TPB2_ENTT) {   // (ENTT) a thread per token: its ids row once, every slot's offsets loaded independently
        for (int t = tid; t < M; t += THREADS) {
            const uint4 row = *reinterpret_cast<const uint4*>(T.ids + t * 8);
            const uint32_t rw[4] = {row.x, row.y, row.z, row.w};
            uint32_t id8[8];
#pragma unroll
            for (int k2 = 0; k2 < 8; ++k2) id8[k2] = (rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu;
            const int w = t >> 5;
            const uint32_t below = (1u << (t & 31)) - 1u;
#pragma unroll
            for (int k = 0; k < TOPK_MAX; ++k) {
                if (k >= topk) break;
                const uint32_t e = id8[k];
                if (e == 0xFFFFu) continue;
                unsigned km = 0;
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) km |= (id8[k2] == e ? 1u : 0u) << k2;
                if (km & ((1u << k) - 1u)) continue;
                const int oe = offe[e];
                const int idx = oe + pre[w * NEXP + e] + __popc(bits[w * NEXP + e] & below);
                T.ent[idx] = static_cast<uint16_t>(t | (km << 9));
                const int j = idx - oe;
                T.tke[t * 8 + k] = static_cast<uint16_t>(((pide[e] + j / NMAX) << 5) | (j % NMAX));
            }
        }
    } else
    for (int p = tid; p < MT; p += THREADS) {
        const int t = p / topk, k = p - t * topk;
        const uint4 row = *reinterpret_cast<const uint4*>(T.ids + t * 8);
        const uint32_t rw[4] = {row.x, row.y, row.z, row.w};
        const int e = static_cast<int>((rw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu);
        if (e == 0xFFFF) continue;
        unsigned km = 0;
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) km |= (((rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu) == static_cast<unsigned>(e) ? 1u : 0u) << k2;
        if (km & ((1u << k) - 1u)) continue;
        const int w = t >> 5;
        const int idx = offe[e] + pre[w * NEXP + e] + __popc(bits[w * NEXP + e] & ((1u << (t & 31)) - 1u));
        T.ent[idx] = static_cast<uint16_t>(t | (km << 9));
        const int j = idx - offe[e];
        T.tke[t * 8 + k] = static_cast<uint16_t>(((pide[e] + j / NMAX) << 5) | (j % NMAX));
    }
    __syncthreads();
}

// TPB2_TABW: the same tables for M <= 8 with the small fused gate (ids in smem, M * topk <= 64 entries, one panel per expert) by warp 0 alone,
// without the CTA-wide phases (cold code + 5 __syncthreads on the prologue's critical path).  Entries p = lane, lane + 32.
__device__ void build_tables_warp(const Params& P, uint8_t* smem, const Tabs& T, const int* gids, int lane) {
    const int M = pM(P), topk = pTK(P), MT = M * topk;
    uint32_t* mask = reinterpret_cast<uint32_t*>(smem + PRO_BITS);   // 256-bit present-expert mask
    if (lane < 8) mask[lane] = 0u;
    __syncwarp();
    int ep[2], tp[2], kp[2];
    unsigned km[2];
    bool first[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int pidx = lane + 32 * h;
        ep[h] = pidx < MT ? gids[pidx] : -1;
        tp[h] = pidx < MT ? pidx / topk : 0;
        kp[h] = pidx - tp[h] * topk;
        km[h] = 0u;
        const bool valid = static_cast<unsigned>(ep[h]) < static_cast<unsigned>(NEXP);
        if (valid) {
            for (int k2 = 0; k2 < topk; ++k2) km[h] |= (gids[tp[h] * topk + k2] == ep[h] ? 1u : 0u) << k2;
            atomicOr(&mask[ep[h] >> 5], 1u << (ep[h] & 31));
        }
        first[h] = valid && (km[h] & ((1u << kp[h]) - 1u)) == 0u;
    }
    const unsigned F0 = __ballot_sync(0xffffffffu, first[0]), F1 = __ballot_sync(0xffffffffu, first[1]);
    __syncwarp();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        if (!first[h]) continue;
        const int e = ep[h], t = tp[h];
        int idx_in = 0, n = 0, ebase = 0;   // rank among this expert's tokens, its token count, its first entry
        for (int q = 0; q < MT; ++q) {
            const bool fq = q < 32 ? ((F0 >> q) & 1u) : ((F1 >> (q - 32)) & 1u);
            if (!fq) continue;
            const int eq = gids[q];
            if (eq == e) { ++n; if (q < t * topk) ++idx_in; }
            else if (eq < e) ++ebase;
        }
        int pe = __popc(mask[e >> 5] & ((1u << (e & 31)) - 1u));
        for (int w = 0; w < (e >> 5); ++w) pe += __popc(mask[w]);
        T.ent[ebase + idx_in] = static_cast<uint16_t>(t | (km[h] << 9));
        if (idx_in == 0) T.pinf[pe] = static_cast<uint32_t>(e) | (static_cast<uint32_t>(n) << 8) | (static_cast<uint32_t>(ebase) << 14);
    }
    if (lane == 0) {
        int U = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) U += __popc(mask[w]);
        T.misc[0] = U; T.misc[1] = __popc(F0) + __popc(F1); T.misc[2] = U;
    }
}

// ------------------------------------------------------------------------------------------------ work
// FC1: item i = panel (i >> 3), tile (i & 7); stage s of 16 (chunks 2 s, 2 s + 1).  CTA b's stream-K range: global stages
// [total b / G, total (b + 1) / G), total = panels x 8 x 16.  Ring entries: x = kind (1 FC1 piece, 2 FC2 unit, -1 end),
// FC1: y = item, z = s0, w = s1;  FC2: y = unit (row tile major: r = unit / panels, panel = unit % panels).
__device__ __forceinline__ int fc1_lo(int total, int b, int G) {   // (32-bit: total * b < 2^32 for MAXP panels; 64-bit division is a slow subroutine)
    return static_cast<int>((static_cast<unsigned>(total) * static_cast<unsigned>(b)) / static_cast<unsigned>(G));
}
__device__ __forceinline__ void panel_of(const Tabs& T, int p, int& e, int& n, int& ebase) {
    const uint32_t v = T.pinf[p];
    e = static_cast<int>(v & 255u); n = static_cast<int>((v >> 8) & 63u); ebase = static_cast<int>(v >> 14);
}
__device__ __forceinline__ float4* part_ptr(const Params& P, int cta) { return reinterpret_cast<float4*>(P.part + static_cast<size_t>(cta) * NMAX * 128); }

// ------------------------------------------------------------------------------------------------ tail (tp4moe2)
__device__ __forceinline__ size_t rs_idx(int src, int lrt, int t, int m) { return (static_cast<size_t>((src * LRT + lrt) * MAXM + t)) * NMSG + m; }
__device__ __forceinline__ size_t ag_idx(int rtile, int t, int m) { return (static_cast<size_t>(rtile * MAXM + t)) * NMSG + m; }
constexpr int OS_MAXM = MAXM / NDEV;
__device__ __forceinline__ size_t os_idx(int src, int rtile, int t, int m) { return (static_cast<size_t>((src * NRT + rtile) * OS_MAXM + t)) * NMSG + m; }
constexpr size_t RS_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;
constexpr int OUTP_LD = DIM + TPB2_OUTPAD;   // outp [MAXM][TOPK_MAX][OUTP_LD]
// T1LL: LL images in P.tcnt past its 2 MAXM NRT counters -- per token (TOPK_MAX slots + the shared output) x 32 row tiles x LLM messages
constexpr int LLM = 22;                                            // LL messages per 128-row tile: 21 x 6 rows + 1 x 2
constexpr size_t LL_OFF = 2 * static_cast<size_t>(MAXM) * NRT;     // (ints; 16-B aligned)
constexpr int LL_TOK = (TOPK_MAX + 1) * 32 * LLM;                  // uint4 per token
constexpr size_t LL_INTS = TPB2_T1LL_MAXM > 0 ? static_cast<size_t>(TPB2_T1LL_MAXM) * LL_TOK * 4 : 0;
static_assert(LL_OFF % 4 == 0, "T1LL alignment");
__device__ __forceinline__ void st_llg(uint4* p, uint4 v) { asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ uint4 ld_llg(const uint4* p) { uint4 v; asm volatile("ld.relaxed.gpu.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ bool fc2l_on(const Params& P) { return TPB2_FC2L && pM(P) == 1 && pSHW(P) && pOS(P) && pND(P) > 1; }
__device__ __forceinline__ bool t1l_on(int M) { return (TPB2_T1L_MAXM && M <= TPB2_T1L_MAXM) || (TPB2_T1L_MINM && M >= TPB2_T1L_MINM && M <= 128); }
__device__ __forceinline__ bool t1ll_on(const Params& P) { return TPB2_T1LL_MAXM && pM(P) <= TPB2_T1LL_MAXM && (pSHW(P) || !pSHD(P)) && pND(P) > 1 && pOS(P); }
constexpr size_t AG_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;

// T1 of one (row tile, token) by one warp: lane l owns values rtile * 256 + 8 l .. + 7
struct T1In { uint4 v[TOPK_MAX]; float wk[TOPK_MAX]; int idk[TOPK_MAX]; uint4 sh; };
// T1 of one (row tile, token) by one warp, in two halves so that a warp's two pairs share one load round trip: lane l owns values
// rtile * 256 + 8 l .. + 7
__device__ __forceinline__ void t1_load(const Params& P, const Tabs& T, bool small_rw, int rtile, int t, int lane, T1In& in) {
    const int topk = pTK(P);
    const int c0 = rtile * 256 + 8 * lane;
    // every load issued at once: slots past topk / invalid ids are loaded but masked off; small fused gate: ids / weights from smem
    in.sh = make_uint4(0u, 0u, 0u, 0u);
    const __nv_bfloat16* shp = pSHW(P) ? P.shout : pSHD(P);
    if (shp && !(TPB2_PROBE & 2048)) in.sh = ld_v4_last_use(shp + static_cast<size_t>(t) * DIM + c0);
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        if (k < topk) {
            in.idk[k] = small_rw ? static_cast<int>(T.sids[t * topk + k]) : __ldcg(P.ids + t * topk + k);
            in.wk[k] = small_rw ? T.srw[t * topk + k] : __ldcg(P.rw + t * topk + k);
            in.v[k] = (TPB2_PROBE & 2048) ? make_uint4(0u, 0u, 0u, 0u) : ld_v4_last_use(P.outp + (static_cast<size_t>(t) * TOPK_MAX + k) * OUTP_LD + c0);
        } else {
            in.idk[k] = -1; in.wk[k] = 0.f; in.v[k] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}
constexpr uint32_t LLS_S = 0xFFFFFFFFu;   // (LLS) sentinel of an empty message slot (word w)
__device__ __forceinline__ uint4 lls_msg(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return make_uint4(x, y, z, w == LLS_S ? 0x7FC07FC0u : w); }
__device__ __forceinline__ void lls_acc(const uint4& v, float* a, bool first) {
    const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[2 * j] = first ? bf16lo(w4[j]) : __fadd_rn(a[2 * j], bf16lo(w4[j]));
        a[2 * j + 1] = first ? bf16hi(w4[j]) : __fadd_rn(a[2 * j + 1], bf16hi(w4[j]));
    }
}
__device__ __forceinline__ uint4 lls_pack(const float* a) {
    return make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));
}
// (LLS) the all-reduce of the T1 rows: one-shot (every source -> every rank) or RS (row-tile owner rtile / LRT) + AG (owner -> every rank)
__device__ __noinline__ void lls_allreduce(const Params& P, int b, int G, int M, int tid, unsigned epoch) {
    if (TPB2_GRID && gridDim.x != static_cast<unsigned>(TPB2_GRID)) __trap();   // (LLS: one reader per slot; another grid would re-read)
    const uint4 S4 = make_uint4(LLS_S, LLS_S, LLS_S, LLS_S);
    const int rk = pRK(P);
    uint4* rs_me = P.rs[rk] + (epoch & 1u) * RS_PAR;
    if (pOS(P)) {
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        for (int x = tid; x < mine * 32; x += THREADS) {
            const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            uint4 v[NDEV];
            for (;;) {
#pragma unroll
                for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
                bool ok = true;
#pragma unroll
                for (int d = 0; d < NDEV; ++d) ok &= v[d].w != LLS_S;
                if (ok) break;
            }
            float a[8];
#pragma unroll
            for (int d = 0; d < NDEV; ++d) lls_acc(v[d], a, d == 0);
            *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m) = lls_pack(a);
#pragma unroll
            for (int d = 0; d < NDEV; ++d) *(rs_me + os_idx(d, rtile, t, m)) = S4;
        }
        return;
    }
    {   // T2: owner of row tiles rk LRT .. reduces (lrt, t), pushes to every rank (self included)
        const int npair = LRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        for (int x = tid; x < mine * 32; x += THREADS) {
            const int pi = x >> 5, m = x & 31, j = b + pi * G, lrt = j / M, t = j - lrt * M;
            uint4 v[NDEV];
            for (;;) {
#pragma unroll
                for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m));
                bool ok = true;
#pragma unroll
                for (int d = 0; d < NDEV; ++d) ok &= v[d].w != LLS_S;
                if (ok) break;
            }
            float a[8];
#pragma unroll
            for (int d = 0; d < NDEV; ++d) lls_acc(v[d], a, d == 0);
            const uint4 o = lls_pack(a), mo = lls_msg(o.x, o.y, o.z, o.w);
            const int rtile = rk * LRT + lrt;
#pragma unroll
            for (int d = 0; d < NDEV; ++d) st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ag_idx(rtile, t, m), mo);
#pragma unroll
            for (int d = 0; d < NDEV; ++d) *(rs_me + rs_idx(d, lrt, t, m)) = S4;
        }
    }
    {   // T3: every (row tile, token) from this rank's AG buffer, TPB2_LLS_RB polls in flight per thread
        uint4* ag_me = P.ag[rk] + (epoch & 1u) * AG_PAR;
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0, tot = mine * 32;
        constexpr int RB = 4;
        for (int x0 = tid; x0 < tot; x0 += RB * THREADS) {
            uint4 v[RB];
            uint4* pm[RB];
            size_t po[RB];
#pragma unroll
            for (int q = 0; q < RB; ++q) {
                const int x = x0 + q * THREADS;
                pm[q] = nullptr; po[q] = 0; v[q] = S4;
                if (x >= tot) continue;
                const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / M, t = j - rtile * M;
                pm[q] = ag_me + ag_idx(rtile, t, m);
                po[q] = static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m;
                v[q] = ld_ll(pm[q]);
            }
#pragma unroll
            for (int q = 0; q < RB; ++q) {
                if (!pm[q]) continue;
                while (v[q].w == LLS_S) v[q] = ld_ll(pm[q]);
                *reinterpret_cast<uint4*>(P.out + po[q]) = v[q];
                *pm[q] = S4;
            }
        }
    }
}
__device__ __forceinline__ void t1_emit(const Params& P, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane, const T1In& in, Xt* xt,
                                        bool noself = false) {   // (noself: this warp receives the pair itself, its own copy stays in stg)
    unsigned on = 0u;
    float wk[TOPK_MAX];
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        if (static_cast<unsigned>(in.idk[k]) < static_cast<unsigned>(NEXP)) on |= 1u << k;
        wk[k] = bf16r(in.wk[k]);   // (PackTopkIds: bf16 routing weights)
    }
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        if (!((on >> k) & 1u)) continue;
        const uint32_t w4[4] = {in.v[k].x, in.v[k].y, in.v[k].z, in.v[k].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[2 * j] = __fmaf_rn(bf16lo(w4[j]), wk[k], acc[2 * j]);
            acc[2 * j + 1] = __fmaf_rn(bf16hi(w4[j]), wk[k], acc[2 * j + 1]);
        }
    }
    const bool shp = (pSHW(P) ? static_cast<const void*>(P.shout) : static_cast<const void*>(pSHD(P))) != nullptr;
    const uint32_t s4[4] = {in.sh.x, in.sh.y, in.sh.z, in.sh.w};
    uint32_t o4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float r0 = bf16r(acc[2 * j]), r1 = bf16r(acc[2 * j + 1]);
        const float y0 = shp ? __fmaf_rn(P.factor, r0, bf16lo(s4[j])) : __fmul_rn(P.factor, r0);
        const float y1 = shp ? __fmaf_rn(P.factor, r1, bf16hi(s4[j])) : __fmul_rn(P.factor, r1);
        o4[j] = pack_bf16x2(y0, y1);
    }
    if (TPB2_LLS && pND(P) > 1) {   // (LLS) lane l's 16 B = message l
        const uint4 msg = lls_msg(o4[0], o4[1], o4[2], o4[3]);
        if (pOS(P)) {
#pragma unroll
            for (int d = 0; d < NDEV; ++d) st_ll(P.rs[(pRK(P) + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, lane), msg);
        } else {
            st_ll(P.rs[TPB2_PROBE_T1LOCAL ? pRK(P) : rtile / LRT] + (epoch & 1u) * RS_PAR + rs_idx(pRK(P), rtile % LRT, t, lane), msg);
        }
        return;
    }
    *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    __syncwarp();
    if (xt && lane == 0) { xt->rec(XT_END(56)); xt->rec(XT_BEG(57)); }
    for (int m = lane; m < NMSG; m += 32) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
        const bool full6 = m < NMSG - 1;
        const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
        if (pND(P) == 1) {
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
            o[0] = msg.x; o[1] = msg.y;
            if (full6) o[2] = msg.z;
        } else if (pOS(P)) {
            if (TPB2_MC & 1) st_mc(P.rsmc + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, m), msg);
            else if (TPB2_MC & 4) {   // multicast to all + this rank's own copy by a local store (it lands at once; the switch's copy
                                      // of the same bytes lands later, harmlessly)
                st_mc(P.rsmc + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, m), msg);
                if (!noself) st_ll(P.rs[pRK(P)] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, m), msg);
            } else {
#pragma unroll
                for (int d = noself ? 1 : 0; d < NDEV; ++d) st_ll(P.rs[(pRK(P) + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, m), msg);
            }
        } else {
            st_ll(P.rs[TPB2_PROBE_T1LOCAL ? pRK(P) : rtile / LRT] + (epoch & 1u) * RS_PAR + rs_idx(pRK(P), rtile % LRT, t, m), msg);
        }
    }
    __syncwarp();
    if (xt && lane == 0) xt->rec(XT_END(57));
}
__device__ __forceinline__ void t1_pair(const Params& P, const Tabs& T, bool small_rw, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane,
                                        Xt* xt = nullptr) {
    if (xt && lane == 0) xt->rec(XT_BEG(56));
    T1In in;
    t1_load(P, T, small_rw, rtile, t, lane, in);
    t1_emit(P, rtile, t, epoch, stg, lane, in, xt);
}

// ARV 16 (one-shot, TCH <= warps): the T1 warp receives its own pair right after its sends -- own copy from stg (no local store / reload), the three peers' messages of lanes l, l + 32 polled in flight together, the
// same rank-order fp32 sum as the one-shot receive
__device__ __forceinline__ void os_recv_own(const Params& P, int rtile, int t, unsigned epoch, const __nv_bfloat16* stg, int lane) {
    const uint4* rs_me = P.rs[pRK(P)] + (epoch & 1u) * RS_PAR;
    const int me = pRK(P), m0 = lane, m1 = lane + 32;
    const bool h1 = m1 < NMSG;
    uint4 v0[NDEV], v1[NDEV];
    unsigned p0 = ((1u << NDEV) - 1u) & ~(1u << me), p1 = h1 ? p0 : 0u;
    while (p0 | p1) {
        const unsigned q0 = p0, q1 = p1;
#pragma unroll
        for (int d = 0; d < NDEV; ++d) if (q0 & (1u << d)) v0[d] = ld_ll(rs_me + os_idx(d, rtile, t, m0));
#pragma unroll
        for (int d = 0; d < NDEV; ++d) if (q1 & (1u << d)) v1[d] = ld_ll(rs_me + os_idx(d, rtile, t, m1));
#pragma unroll
        for (int d = 0; d < NDEV; ++d) {
            if ((q0 & (1u << d)) && v0[d].w == epoch) p0 &= ~(1u << d);
            if ((q1 & (1u << d)) && v1[d].w == epoch) p1 &= ~(1u << d);
        }
    }
    const uint32_t* w0 = reinterpret_cast<const uint32_t*>(stg + 6 * m0);
    const uint32_t* w1 = reinterpret_cast<const uint32_t*>(stg + 6 * (h1 ? m1 : m0));
    const uint4 o0 = make_uint4(w0[0], w0[1], w0[2], epoch), o1 = make_uint4(w1[0], w1[1], m1 < NMSG - 1 ? w1[2] : 0u, epoch);
#pragma unroll
    for (int d = 0; d < NDEV; ++d) if (d == me) { v0[d] = o0; v1[d] = o1; }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        if (h == 1 && !h1) break;
        const uint4* v = h ? v1 : v0;
        const int m = h ? m1 : m0;
        float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
        for (int d = 1; d < NDEV; ++d) {
            a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
            a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
            a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
        }
        uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
        o[0] = pack_bf16x2(a[0], a[1]); o[1] = pack_bf16x2(a[2], a[3]);
        if (m < NMSG - 1) o[2] = pack_bf16x2(a[4], a[5]);
    }
}

// T1LL: (row tile rtile, token t) from the LL images (small fused gate: ids / weights in smem): lane l polls messages l, l + 32 of the 44 per
// contribution (2 x 128-row tiles) until every tag is this epoch, then the T1 sum exactly as t1_emit (slots in order, bf16 rounding, factor *
// r + shared) into stg[256] and the one-shot LL sends
__device__ __noinline__ void t1ll_pair(const Params& P, const Tabs& T, int rtile, int t, unsigned epoch, __nv_bfloat16* stg, int lane, Xt* xt) {
    const uint4* ll = reinterpret_cast<const uint4*>(P.tcnt + LL_OFF) + static_cast<size_t>(t) * LL_TOK;
    const int topk = pTK(P);
    unsigned on = 0u;
    float wk[TOPK_MAX];
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        const int id = k < topk ? static_cast<int>(T.sids[t * topk + k]) : -1;
        if (static_cast<unsigned>(id) < static_cast<unsigned>(NEXP)) on |= 1u << k;
        wk[k] = k < topk ? bf16r(T.srw[t * topk + k]) : 0.f;
    }
    if (pSHW(P)) on |= 1u << TOPK_MAX;   // (the fused shared expert's output: LL slot TOPK_MAX)
    uint4 mv[TOPK_MAX + 1][2];
    const bool two = lane + 32 < 2 * LLM;
    if (xt && lane == 0) xt->rec(XT_BEG(51), static_cast<uint32_t>(rtile));
    for (;;) {
        bool ok = true;
#pragma unroll
        for (int k = 0; k <= TOPK_MAX; ++k) {
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = lane + 32 * jj, h = j >= LLM ? 1 : 0, c = j - h * LLM;
                if (((on >> k) & 1u) && (jj == 0 || two)) {
                    mv[k][jj] = ld_llg(ll + (k * 32 + 2 * rtile + h) * LLM + c);
                    ok &= mv[k][jj].w == epoch;
                } else {
                    mv[k][jj] = make_uint4(0u, 0u, 0u, epoch);
                }
            }
        }
        if (__all_sync(0xffffffffu, ok)) break;
        __nanosleep(TPB2_T1LL_SLEEP);
    }
    if (xt && lane == 0) { xt->rec(XT_END(51)); xt->rec(XT_BEG(56)); }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int j = lane + 32 * jj, h = j >= LLM ? 1 : 0, c = j - h * LLM;
        if (jj == 1 && !two) break;
        float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < TOPK_MAX; ++k) {
            if (!((on >> k) & 1u)) continue;
            const uint32_t w3[3] = {mv[k][jj].x, mv[k][jj].y, mv[k][jj].z};
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                acc[2 * q] = __fmaf_rn(bf16lo(w3[q]), wk[k], acc[2 * q]);
                acc[2 * q + 1] = __fmaf_rn(bf16hi(w3[q]), wk[k], acc[2 * q + 1]);
            }
        }
        const uint32_t s3[3] = {mv[TOPK_MAX][jj].x, mv[TOPK_MAX][jj].y, mv[TOPK_MAX][jj].z};
        uint32_t o3[3];
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            const float r0 = bf16r(acc[2 * q]), r1 = bf16r(acc[2 * q + 1]);
            const float y0 = pSHW(P) ? __fmaf_rn(P.factor, r0, bf16lo(s3[q])) : __fmul_rn(P.factor, r0);
            const float y1 = pSHW(P) ? __fmaf_rn(P.factor, r1, bf16hi(s3[q])) : __fmul_rn(P.factor, r1);
            o3[q] = pack_bf16x2(y0, y1);
        }
        uint32_t* sw = reinterpret_cast<uint32_t*>(stg + h * 128 + 6 * c);
        sw[0] = o3[0];
        if (c < LLM - 1) { sw[1] = o3[1]; sw[2] = o3[2]; }
    }
    __syncwarp();
    if (xt && lane == 0) { xt->rec(XT_END(56)); xt->rec(XT_BEG(57)); }
    for (int m = lane; m < NMSG; m += 32) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
        const bool full6 = m < NMSG - 1;
        const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
#pragma unroll
        for (int d = 0; d < NDEV; ++d) st_ll(P.rs[(pRK(P) + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), rtile, t, m), msg);
    }
    __syncwarp();
    if (xt && lane == 0) xt->rec(XT_END(57));
}

__device__ void tail(const Params& P, const Tabs& T, bool small_rw, int tid, unsigned epoch, unsigned par, uint8_t* smem, Xt& xt) {
    if (TPB2_PROBE_END) return;
    const int G = TPB2_GRID ? TPB2_GRID : static_cast<int>(gridDim.x), b = blockIdx.x, M = pM(P);
    const unsigned need = static_cast<unsigned>(T.misc[2] + T.misc[4]) * 2u;   // FC2 row tiles (routed + shared panels) per 256-row tile
    const int TCH = (M + max(1, G / NRT) - 1) / max(1, G / NRT);
    const int ntc = (M + TCH - 1) / TCH, nitem = NRT * ntc;
    volatile int* s_item = reinterpret_cast<volatile int*>(T.misc + 8);
    unsigned done_rt = 0u, done_it = 0u;
    // ARV 16 (one-shot, every item's tokens on distinct warps: TCH <= warps): each pair is received on every rank by the warp that T1'd
    // it there, right after its sends (os_recv_own); the CTA-wide receive below is skipped
    const bool fused = (TPB2_ARV & 16) && TCH <= THREADS / 32 && pND(P) > 1 && pOS(P) && !fc2l_on(P) && !TPB2_PROBE_T1LOCAL;
    if (TPB2_T1W && TPB2_SFCP_MMA) {   // the pairs warp 7 left open: one warp each, waiting for its FC2 count
        const unsigned long long left = static_cast<unsigned long long>(static_cast<unsigned>(T.misc[12])) |
                                        (static_cast<unsigned long long>(static_cast<unsigned>(T.misc[13])) << 32);
        const int warp = tid >> 5, lane = tid & 31;
        __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem) + warp * 256;   // (A ring: dead)
        int rank = 0;
        for (unsigned long long gm = left; gm; gm &= gm - 1ull, ++rank) {
            if (rank % (THREADS / 32) != warp) continue;
            const int k = __ffsll(static_cast<long long>(gm)) - 1, j = b + k * G, t = j % M, rt = j / M;
            unsigned nv = 0u;
            for (int s2 = 0; s2 < pTK(P); ++s2) nv += static_cast<unsigned>(__ldcg(P.ids + t * pTK(P) + s2)) < static_cast<unsigned>(NEXP) ? 1u : 0u;
            if (lane == 0) wait_ge(P.tcnt + par * MAXM * NRT + t * NRT + rt, 2u * (nv + (pSHW(P) ? 1u : 0u)));
            __syncwarp();
            fence_acq_rel();
            t1_pair(P, T, small_rw, rt, t, epoch, stg, lane);
        }
        __syncthreads();
    }
    for (;;) {
        if (TPB2_T1W && TPB2_SFCP_MMA) break;   // (T1: warp 7 during FC2 + the loop above)
        if (t1l_on(M) || fc2l_on(P)) break;   // (T1L: every pair T1'd by its last contributor's warp 7; FC2L: by the FC2 CTA)
        if (t1ll_on(P)) break;                             // (T1LL: every pair T1'd by warp 7 of CTA j)
        if (tid == 0) {
            const int it = TPB2_T1STATIC ? b + static_cast<int>(done_it++) * G : static_cast<int>(atomicAdd(P.fc2_next + 2 + par, 1u));
            if (it < nitem) {
                const int rt = it / ntc;
                xt.rec(XT_BEG(51), static_cast<uint32_t>(it));
                if (!((done_rt >> rt) & 1u)) { wait_ge(P.rt_cnt + par * NRT + rt, need); done_rt |= 1u << rt; }
                xt.rec(XT_END(51));
            }
            *s_item = it;
        }
        __syncthreads();
        const int it = *s_item;
        __syncthreads();
        if (it >= nitem) break;
        if (tid == 0) xt.rec(XT_BEG(52), static_cast<uint32_t>(it));
        const int rtile = it / ntc, t0 = (it - rtile * ntc) * TCH, nt = min(TCH, M - t0);
        __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem) + (tid >> 5) * 256;
        constexpr int NW = THREADS / 32;
        for (int tt = tid >> 5; tt < nt; tt += 2 * NW) {   // (a warp's pairs tt, tt + NW: both pairs' loads in one round trip)
            Xt* xp = tid < 32 ? &xt : nullptr;
            if (xp && (tid & 31) == 0) xp->rec(XT_BEG(56));
            T1In ia, ib;
            t1_load(P, T, small_rw, rtile, t0 + tt, tid & 31, ia);
            const bool two = tt + NW < nt;
            if (two) t1_load(P, T, small_rw, rtile, t0 + tt + NW, tid & 31, ib);
            t1_emit(P, rtile, t0 + tt, epoch, stg, tid & 31, ia, xp, fused);
            if (two) t1_emit(P, rtile, t0 + tt + NW, epoch, stg, tid & 31, ib, nullptr);
            if (fused) os_recv_own(P, rtile, t0 + tt, epoch, stg, tid & 31);   // (TCH <= warps: never two)
        }
        if (tid == 0) xt.rec(XT_END(52));
    }
    if (DBGP && tid == 0) DBGP[b * 32 + 6] = gtime();
    if (tid == 0) xt.rec(XT_BEG(55));
    if (pND(P) == 1) return;
    if (TPB2_PROBE_T1LOCAL) return;   // (probe: no all-reduce -- wrong output)
    if (fused) return;
    if (TPB2_LLS) {   // (LLS) sentinel messages: their own receive / reduce / push loops
        static_assert(!TPB2_LLS || (!(TPB2_ARV & 16) && !TPB2_MC && !TPB2_FC2L && !TPB2_T1LL_MAXM && !TPB2_T1L_MAXM && !TPB2_T1L_MINM && !TPB2_T1W),
                      "LLS: the plain T1 + one-shot / RS + AG tail");
        lls_allreduce(P, b, G, M, tid, epoch);
        return;
    }
    if (pOS(P)) {
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* rs_me = P.rs[pRK(P)] + (epoch & 1u) * RS_PAR;
        const bool hl = fc2l_on(P);   // (FC2L: 2 x 22 messages per 256-row tile, numbered per 128-row half)
        const int nm = hl ? 2 * 22 : NMSG;
        for (int x = tid; x < mine * nm; x += THREADS) {
            const int pi = x / nm, m = x - pi * nm, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            uint4 v[NDEV];
            unsigned pending = (1u << NDEV) - 1u;
            if (TPB2_ARV & 8) {
                do {
                    const unsigned p0 = pending;
#pragma unroll
                    for (int d = 0; d < NDEV; ++d) if (p0 & (1u << d)) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
#pragma unroll
                    for (int d = 0; d < NDEV; ++d) if ((p0 & (1u << d)) && v[d].w == epoch) pending &= ~(1u << d);
                } while (pending);
            } else
            do {
#pragma unroll
                for (int d = 0; d < NDEV; ++d)
                    if (pending & (1u << d)) { v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m)); if (v[d].w == epoch) pending &= ~(1u << d); }
            } while (pending);
            float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
            for (int d = 1; d < NDEV; ++d) {
                a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
                a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
                a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
            }
            const int row0 = hl ? (m / 22) * 128 + 6 * (m % 22) : 6 * m, nw = hl ? (m % 22 == 21 ? 1 : 3) : (m < NMSG - 1 ? 3 : 2);
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + row0);
            o[0] = pack_bf16x2(a[0], a[1]);
            if (nw > 1) o[1] = pack_bf16x2(a[2], a[3]);
            if (nw > 2) o[2] = pack_bf16x2(a[4], a[5]);
        }
        return;
    }
    if (TPB2_ARV & 2) {   // T2 + T3 in one poll loop: a thread's <= 1 T2 message (4 sources) and <= 4 T3 messages in flight together
        const int n2 = (b < LRT * M ? (LRT * M - b + G - 1) / G : 0) * NMSG, n3 = (b < NRT * M ? (NRT * M - b + G - 1) / G : 0) * NMSG;
        if (n2 <= THREADS && n3 <= 4 * THREADS) {
            const uint4* rs_me = P.rs[pRK(P)] + (epoch & 1u) * RS_PAR;
            const uint4* ag_me = P.ag[pRK(P)] + (epoch & 1u) * AG_PAR;
            unsigned p2 = 0u, p3 = 0u;
            size_t a2 = 0, a3[4];
            int lrt = 0, t2 = 0, m2 = 0;
            if (tid < n2) {
                const int pi = tid / NMSG, j = b + pi * G;
                m2 = tid - pi * NMSG; lrt = j / M; t2 = j - lrt * M;
                a2 = rs_idx(0, lrt, t2, m2); p2 = (1u << NDEV) - 1u;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int x = tid + k * THREADS;
                a3[k] = 0;
                if (x < n3) {
                    const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
                    a3[k] = ag_idx(rtile, t, m); p3 |= 1u << k;
                }
            }
            constexpr size_t RS_SRC = static_cast<size_t>(LRT) * MAXM * NMSG;   // rs_idx(d + 1, ...) - rs_idx(d, ...)
            uint4 v2[NDEV], v3[4];
            while (p2 | p3) {
                const unsigned q2 = p2, q3 = p3;
#pragma unroll
                for (int d = 0; d < NDEV; ++d) if (q2 & (1u << d)) v2[d] = ld_ll(rs_me + a2 + d * RS_SRC);
#pragma unroll
                for (int k = 0; k < 4; ++k) if (q3 & (1u << k)) v3[k] = ld_ll(ag_me + a3[k]);
#pragma unroll
                for (int d = 0; d < NDEV; ++d) if ((q2 & (1u << d)) && v2[d].w == epoch) p2 &= ~(1u << d);
                if (q2 && !p2) {   // all four sources in: reduce in rank order, all-gather push
                    float a[6] = {bf16lo(v2[0].x), bf16hi(v2[0].x), bf16lo(v2[0].y), bf16hi(v2[0].y), bf16lo(v2[0].z), bf16hi(v2[0].z)};
#pragma unroll
                    for (int d = 1; d < NDEV; ++d) {
                        a[0] = __fadd_rn(a[0], bf16lo(v2[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v2[d].x));
                        a[2] = __fadd_rn(a[2], bf16lo(v2[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v2[d].y));
                        a[4] = __fadd_rn(a[4], bf16lo(v2[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v2[d].z));
                    }
                    const uint4 mo = make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), epoch);
                    const size_t ai = ag_idx(pRK(P) * LRT + lrt, t2, m2);
                    if (TPB2_MC & 2) st_mc(P.agmc + (epoch & 1u) * AG_PAR + ai, mo);
                    else {
#pragma unroll
                        for (int d = 0; d < NDEV; ++d) st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ai, mo);
                    }
                }
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    if ((q3 & (1u << k)) && v3[k].w == epoch) {
                        p3 &= ~(1u << k);
                        const int x = tid + k * THREADS, pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
                        uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
                        o[0] = v3[k].x; o[1] = v3[k].y;
                        if (m < NMSG - 1) o[2] = v3[k].z;
                    }
            }
            return;
        }
    }
    {   // T2: owner reduce in rank order, all-gather push
        const int npair = LRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* rs_me = P.rs[pRK(P)] + (epoch & 1u) * RS_PAR;
        for (int x = tid; x < mine * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, lrt = j / M, t = j - lrt * M;
            uint4 v[NDEV];
            unsigned pending = (1u << NDEV) - 1u;
            if (TPB2_ARV & 8) {
                do {
                    const unsigned p0 = pending;
#pragma unroll
                    for (int d = 0; d < NDEV; ++d) if (p0 & (1u << d)) v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m));
#pragma unroll
                    for (int d = 0; d < NDEV; ++d) if ((p0 & (1u << d)) && v[d].w == epoch) pending &= ~(1u << d);
                } while (pending);
            } else
            do {
#pragma unroll
                for (int d = 0; d < NDEV; ++d)
                    if (pending & (1u << d)) { v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m)); if (v[d].w == epoch) pending &= ~(1u << d); }
            } while (pending);
            float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
            for (int d = 1; d < NDEV; ++d) {
                a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
                a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
                a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
            }
            const uint4 mo = make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), epoch);
            const int rtile = pRK(P) * LRT + lrt;
            if (TPB2_MC & 2) st_mc(P.agmc + (epoch & 1u) * AG_PAR + ag_idx(rtile, t, m), mo);
            else {
#pragma unroll
                for (int d = 0; d < NDEV; ++d) st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ag_idx(rtile, t, m), mo);
            }
        }
    }
    {   // T3: pull the reduced rows
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* ag_me = P.ag[pRK(P)] + (epoch & 1u) * AG_PAR;
        if (TPB2_ARV & 1) {   // a thread's (up to 4) messages polled in flight together
            const int tot = mine * NMSG;
            for (int x0 = tid; x0 < tot; x0 += 4 * THREADS) {
                unsigned p3 = 0u;
                size_t a3[4];
                uint4 v3[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int x = x0 + k * THREADS;
                    a3[k] = 0;
                    if (x < tot) {
                        const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
                        a3[k] = ag_idx(rtile, t, m); p3 |= 1u << k;
                    }
                }
                while (p3) {
                    const unsigned q3 = p3;
#pragma unroll
                    for (int k = 0; k < 4; ++k) if (q3 & (1u << k)) v3[k] = ld_ll(ag_me + a3[k]);
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        if ((q3 & (1u << k)) && v3[k].w == epoch) {
                            p3 &= ~(1u << k);
                            const int x = x0 + k * THREADS, pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
                            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
                            o[0] = v3[k].x; o[1] = v3[k].y;
                            if (m < NMSG - 1) o[2] = v3[k].z;
                        }
                }
            }
            return;
        }
        for (int x = tid; x < mine * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            uint4 v;
            do { v = ld_ll(ag_me + ag_idx(rtile, t, m)); } while (v.w != epoch);
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
            o[0] = v.x; o[1] = v.y;
            if (m < NMSG - 1) o[2] = v.z;
        }
    }
}

// TPB2_LEANMMA 3: the MMA issue loop of one accumulator with its first ring slot S0 a template constant: unrolled by NS, every stage's
// slot is a compile-time constant -> smem descriptors / scale / barrier addresses are uniform constants off the smem base (no per-stage
// vector -> uniform conversions; the issuing warp paces the weight stream at ~8 MMAs per stage)
template <int S0, bool SH>
__device__ __forceinline__ void mma_stages(int nst, int dz, uint32_t ph0, uint32_t dacc, uint32_t id0, uint32_t bstep, uint32_t sb, uint32_t tsf0,
                                           uint32_t bfull0, uint32_t bempty0) {
    for (int j0 = 0; j0 < nst; j0 += NS) {
        const uint32_t phb = ph0 ^ ((static_cast<uint32_t>(j0) / NS) & 1u);
#pragma unroll
        for (int u = 0; u < NS; ++u) {
            const int j = j0 + u;
            if (j >= nst) break;
            const int st = (S0 + u) % NS;
            mbar_wait(bfull0 + 8 * st, phb ^ static_cast<uint32_t>(((S0 + u) / NS) & 1));
            tc_fence_after();
            const uint32_t tsf = tsf0 + st * TSF_STRIDE;
            const uint64_t ad0 = desc_sw128(sb + OFF_A + st * A_BYTES), bd0 = desc_sw128(sb + OFF_B + st * B_BYTES);
            const uint32_t first = j != 0 ? 1u : 0u;
            if (elect_one()) {
#pragma unroll
                for (int c = 0; c < KC; ++c) tc_cp_sf(tsf + 4 * c, desc_sf(sb + OFF_SFA + (st * KC + c) * 512));
                tc_cp_sf(tsf + TSF_B, desc_sf(sb + OFF_SFB + st * 512));
                if (SH) {
                    const uint32_t sfb = tsf + TSF_B + 2 * ((dz + j) & 1);
#pragma unroll
                    for (int kb = 0; kb < 4; ++kb) {
                        const uint32_t idk = id0 | (kb << 4) | (kb << 29), acc = kb != 0 ? 1u : first;
                        tc_mma(dacc, ad0 + 2 * kb, bd0 + 2 * kb, idk, tsf, sfb, acc);
                        tc_mma(dacc + 32, ad0 + (16384 >> 4) + 2 * kb, bd0 + 2 * kb, idk, tsf + 4, sfb, acc);
                    }
                } else {
#pragma unroll
                    for (int c = 0; c < KC; ++c) {
                        const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + static_cast<uint64_t>(c * bstep);
#pragma unroll
                        for (int kb = 0; kb < 4; ++kb)
                            tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + TSF_B + 2 * c, (c | kb) != 0 ? 1u : first);
                    }
                }
                tc_commit(bempty0 + 8 * st);
            }
            __syncwarp();
        }
    }
}

// ------------------------------------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(THREADS, 1) tpb2_kernel(const __grid_constant__ Params P) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    uint8_t* smem = smem_raw + (sb - sraw);
    const Tabs T = tabs(smem);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int G = TPB2_GRID ? TPB2_GRID : static_cast<int>(gridDim.x), b = blockIdx.x, M = pM(P);
    const bool f2s = TPB2_FC2STATIC || (M >= TPB2_FC2S_LO && M <= TPB2_FC2S_HI);   // FC2 tickets in static panel order
    const bool fc2l = fc2l_on(P);
    auto bar = [&](int i) { return sb + OFF_TAB + TAB_BAR + 8 * i; };
    if (DBGP && tid == 0) { DBGP[b * 32 + 0] = gtime(); DBGP[b * 32 + 30] = clock64(); }
    if (TPB2_EARLYTRIG) griddep_launch();
    if (TPB2_PROBE_EMPTY) { griddep_wait(); return; }   // (probe: launch + PDL overhead of this kernel alone; wrong output)
    Xt xt;   // (INIT probe: this warp's ring slot + counter in registers)
    xt.init(TPB2_TRACE ? static_cast<void*>(P.dbg_) : nullptr);
    if (tid == 0) xt_anchor(TPB2_TRACE ? static_cast<void*>(P.dbg_) : nullptr, false);
    constexpr int B_FULL = 0, B_EMPTY = NS, B_SFR = 2 * NS, B_ACCF = 3 * NS, B_ACCE = 3 * NS + ACC, B_WF = 3 * NS + 2 * ACC, B_WE = B_WF + WR;
    constexpr int B_FULLP = B_WE + WR, B_MFULL = B_FULLP + NS, B_MEMPTY = B_MFULL + 2, B_RW = B_MEMPTY + 2, B_RX = B_RW + 1, B_RM = B_RX + 1,
                  B_RX1 = B_RM + 1, B_RF0 = B_RX1 + 1, B_RF1 = B_RF0 + 1, B_HF0 = B_RF1 + 1, B_HE0 = B_HF0 + 2;
    // PAIR: cluster rank (0 = leader: scheduler + MMA issuer), scheduling unit pb of PG (unit = the CTA pair), tile of a unit index
    constexpr bool PR = TPB2_PAIR, CS = TPB2_CSPLIT;
    uint32_t crank = 0u;
    if (PR || CS) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    const bool lead = !PR || crank == 0u;   // (PAIR: the scheduler / MMA issuer; CSPLIT: both CTAs schedule their own work)
    const int pb1 = (PR || CS) ? (b >> 1) : b, PG1 = (PR || CS) ? (G >> 1) : G;   // FC1 scheduling units (PAIR / CSPLIT: clusters)
    const int pb = PR ? (b >> 1) : b, PG = PR ? (G >> 1) : G;
    constexpr int NT1U = PR ? NT1 / 2 : NT1, NT1SU = PR ? NT1S / 2 : NT1S, NRT2U = PR ? NRT2 / 2 : NRT2;
    auto titem = [&](int u) { return PR ? (u / NT1U) * NT1 + 2 * (u % NT1U) + static_cast<int>(crank) : u; };     // routed FC1 tile item
    auto titemS = [&](int u) { return PR ? (u / NT1SU) * NT1S + 2 * (u % NT1SU) + static_cast<int>(crank) : u; };  // shared FC1 tile item
    auto trow = [&](int x) { return PR ? 2 * x + static_cast<int>(crank) : x; };                                    // FC2 row tile of a unit row
    auto cta_of = [&](int u) { return PR ? 2 * u + static_cast<int>(crank) : u; };                                  // this rank's CTA of unit u
    auto we_arrive = [&](int slot) {   // ring slot consumed (PAIR: every consumer counts on the leader's barrier; the caller has consumed
                                       // the entry's registers, so a relaxed remote arrive cannot overtake the read)
        if (PR && !lead) mbar_arrive_remote_relaxed(peer_addr(bar(B_WE + slot), 0u)); else mbar_arrive(bar(B_WE + slot));
    };
    auto use4 = [](const int4& d) { asm volatile("" ::"r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w)); };
    // ---- prologue: barriers, TMEM, lane table (before the PDL wait)
    if (tid < NBAR) {   // one barrier per thread: FULL 2 (producer + B), EMPTY / SFR / ACCF / WF 1, ACCE 8 epilogue warps (PAIR leader: 16),
                        // WE consumers (PAIR leader: + the peer's 8 + 3), FULLP 1
        const int i = tid;
        const int nwe = 8 + 2 + (TPB2_SFCP_MMA ? 0 : 1) + (PR ? 8 + 3 : 0);
        const int cnt = i < B_EMPTY ? 2 : (i < B_ACCE ? 1 : (i < B_WF ? (PR ? 16 : 8) : (i < B_WE ? 1 : (i < B_FULLP ? nwe : 1))));
        mbar_init(bar(i), cnt);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (PR || CS) cluster_sync_all();   // (the peer's barriers exist before any remote arrive)
    const int SHQ = (TPB2_SHW1 && M >= TPB2_SHW1) ? 1 : TPB2_SHQ, SHC = NSTG1S / SHQ;   // shared pieces per item, chunks per piece
    const bool shgu = TPB2_SHGU && !PR && !CS && SHQ == 1 && (M <= TPB2_SHGU_MAXM || (M >= TPB2_SHGU_LO2 && M <= TPB2_SHGU_HI2));   // (gate / up chains on two CTAs)
    const int nshq0 = pSHW(P) ? ((M + NMAX - 1) / NMAX) * NT1SU * SHQ * (shgu ? 2 : 1) : 0;   // shared pieces (units; = nshq below)
    const int shh = (shgu && pb < nshq0) ? (pb & 1) : -1;   // (SHGU: this CTA's shared piece is the gate (0) or the up (1) chain)
    const bool shearly = TPB2_SHEARLY && nshq0 > 0 && 2 * nshq0 <= PG && M > TPB2_SHEARLY_OFFM;
    const int npre = (shearly && pb < nshq0) ? min(TPB2_SHPRE, SHC) : 0;   // ring stages issued before the PDL wait
    if (npre) {
        __syncthreads();   // (the barrier inits)
        if (tid == 0) {   // shared quarter b: tile k = (b / 4) % NT1S, chunks 8 (b % 4) ..; the producer skips these stages
            const int k = titemS(shgu ? pb >> 1 : pb / SHQ) % NT1S, c0 = shgu ? 0 : (pb % SHQ) * SHC;
            const uint32_t full_lead = PR ? peer_addr(bar(B_FULL), 0u) : 0u;
            for (int j = 0; j < npre; ++j) {
                if (shh >= 0) {   // (SHGU: chunks 2 j, 2 j + 1 of this CTA's gate or up rows + their scales -- a full 32 KB slot)
                    mbar_expect_tx(bar(B_FULL + j), 2 * 16384 + 2 * 512);
                    for (int h2 = 0; h2 < 2; ++h2) {
                        tma3d(sb + OFF_A + j * A_BYTES + h2 * 16384, &P.map1s, 0, shh * INTER + 128 * k, 2 * j + h2, bar(B_FULL + j), SH_HINT);
                        bulk_g2s(sb + OFF_SFA + j * KC * 512 + h2 * 512, P.sfa1s + (static_cast<size_t>(shh * (INTER / 128) + k) * NCH1 + 2 * j + h2) * 512, 512, bar(B_FULL + j));
                    }
                    continue;
                }
                mbar_expect_tx(bar(B_FULL + j), (PR ? (lead ? 4 * 16384 : 0) : 2 * 16384) + 2 * 512);
                if (PR && !lead) {
                    tma3d_cg2(sb + OFF_A + j * A_BYTES, &P.map1s, 0, 128 * k, c0 + j, full_lead + 8 * j);
                    tma3d_cg2(sb + OFF_A + j * A_BYTES + 16384, &P.map1s, 0, INTER + 128 * k, c0 + j, full_lead + 8 * j);
                } else {
                    tma3d(sb + OFF_A + j * A_BYTES, &P.map1s, 0, 128 * k, c0 + j, bar(B_FULL + j), SH_HINT);
                    tma3d(sb + OFF_A + j * A_BYTES + 16384, &P.map1s, 0, INTER + 128 * k, c0 + j, bar(B_FULL + j), SH_HINT);
                }
                bulk_g2s(sb + OFF_SFA + j * KC * 512, P.sfa1s + (static_cast<size_t>(k) * NCH1 + c0 + j) * 512, 512, bar(B_FULL + j));
                bulk_g2s(sb + OFF_SFA + j * KC * 512 + 512, P.sfa1s + (static_cast<size_t>(INTER / 128 + k) * NCH1 + c0 + j) * 512, 512, bar(B_FULL + j));
            }
            xt.rec(XT_PT(6));
        }
    }
    // in-kernel router unit u = G - 1 - b (part u / 16, experts 16 (u % 16) ..): its W slice (static) loaded before the PDL wait
    const int ru = G - 1 - b;
    const bool rcta = pRS(P) > 0 && ru < 16 * pRS(P) * (TPB2_RRB ? P.rB : 1);
    constexpr uint32_t RW_OFF = 96 * 1024;   // (x rows: [0, rch R 128 + slack); W rows [96 KB, + rch 2 KB): ring A / B stages, dead until the tables)
    if (rcta) {
        __syncthreads();   // (the barrier inits)
        if (tid == 0) {
            mbar_expect_tx(bar(B_RW), static_cast<uint32_t>(P.rch) * 2048u);
            tma3d(sb + RW_OFF, &P.maprw, 0, 16 * (ru % 16), P.rk0[(ru / 16) % pRS(P)] / 64, bar(B_RW));   // (RRB: units ru / 16 / S = row block)
        }
    }
    // in-kernel dsv3 router (rF): expert 3 ru + g on CTA G - 1 - ru, thread group g = warps 4 g .. 4 g + 3 in the dsv3 CTA's layout
    // (128 threads, 8 bf16 per thread per K iteration of 1024); its W row slice (static) in registers before the PDL wait
    const int rfe = 3 * ru + (warp >> 2);
    const bool rfc = pRF(P) && rfe < NEXP;
    uint4 rwv[DIM / 1024];
    if (rfc) {
        const uint4* wr = reinterpret_cast<const uint4*>(P.rwf + static_cast<size_t>(rfe) * DIM) + (tid & 127);
#pragma unroll
        for (int i = 0; i < DIM / 1024; ++i) rwv[i] = __ldg(wr + 128 * i);
    }
    if (TPB2_SHL2 && pSHW(P) && pM(P) <= TPB2_SHL2_MAXM && P.wp[2] && tid == 96) {   // the shared gate_up slice (4 MB, routing-independent) -> L2
        for (int c = b; c < (1024 * 4096) / 65536; c += G) pf_bulk(P.wp[2] + static_cast<size_t>(c) * 65536, 65536);   // (DRAM idles in the prologue)
    }
    if (tid == 64) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(&P.map1) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&P.map2) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&P.mapxq) : "memory");
    }
    if (warp == 1) {
        asm volatile("tcgen05.alloc." TPB2_CG ".sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(T.misc + 3)), "r"(TMEM_COLS) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit." TPB2_CG ".sync.aligned;" ::: "memory");
        tc_fence_before();
    }
    if (tid < 32) reinterpret_cast<uint32_t*>(T.lanes)[tid] = reinterpret_cast<const uint32_t*>(P.lanes)[tid];
    if (DBGP && tid == 0) DBGP[b * 32 + 25] = gtime();
    if (TPB2_GWARM && warp == 3 && pGL(P)) {   // (dummy logits: the bias; results unused)
        if (lane == 0) xt.rec(XT_BEG(7));
        const GateR r = gate_call(P, P.gbias, 1, lane, nullptr);
        asm volatile("" ::"r"(r.id), "f"(r.wt));
        if (lane == 0) xt.rec(XT_END(7));
    }
    if (tid == 0) xt.rec(XT_BEG(2));
    griddep_wait();
    // (ARV 32: epoch = completed calls + 1 from the u64 CTA-exit count work[2..3] (+ G per call), no last-CTA detection at exit)
    const unsigned epoch = (TPB2_ARV & 32) ? static_cast<unsigned>(__ldcg(reinterpret_cast<const unsigned long long*>(P.work + 2)) /
                                                                   static_cast<unsigned long long>(G)) + 1u
                                           : __ldcg(P.work) + 1u, par = epoch & 1u;
    if (DBGP && tid == 0) DBGP[b * 32 + 26] = gtime();
    if (tid == 0) xt.rec(XT_END(2));
    if (TPB2_T1W)   // (every CTA a slice) the next launch's T1 counters
        for (int i = b * THREADS + tid; i < MAXM * NRT; i += G * THREADS) P.tcnt[(par ^ 1u) * MAXM * NRT + i] = 0u;
    if (b == 0) {   // the next launch's parity set (the previous launch, its user, has completed)
        for (int i = tid; i < MAXP; i += THREADS) P.pan_cnt[(par ^ 1u) * MAXP + i] = 0u;
        for (int i = tid; i < MAXP + 32; i += THREADS) P.rdy[(par ^ 1u) * (MAXP + 32) + i] = 0u;
        for (int i = tid; i < MAXM; i += THREADS) P.xflag[(par ^ 1u) * MAXM + i] = 0u;
        if (tid < NRT) P.rt_cnt[(par ^ 1u) * NRT + tid] = 0u;
        if (tid == NRT && !(TPB2_ARV & 32)) P.work[1 + (par ^ 1u)] = 0u;
        if (tid == NRT + 1) { P.fc2_next[par ^ 1u] = 0u; P.fc2_next[2 + (par ^ 1u)] = 0u; P.gcnt[par ^ 1u] = 0u; }
        if (TPB2_LASTM) for (int i = tid; i < 256; i += THREADS) P.tcnt[(par ^ 1u) * MAXM * NRT + i] = 0u;   // (LASTM counters)
        if (tid == 0) P.tcnt[(par ^ 1u) * MAXM * NRT + 300] = 0u;   // (in-kernel router units done)
        if (TPB2_T1L_MAXM || TPB2_T1L_MINM)
            for (int i = tid; i < (TPB2_T1L_MINM ? 128 : TPB2_T1L_MAXM) * NRT; i += THREADS) P.tcnt[(par ^ 1u) * MAXM * NRT + 1024 + i] = 0u;   // (T1L)
    }
    if (rcta && tid == 0) xt.rec(XT_BEG(76));
    if (rcta) {   // ---- in-kernel router unit: x part (rows < M real; rows R..63 of a chunk's 64-row tile alias the next chunk: their D rows are
                  // not used), kind::f16 K16 chain in K order (== the cuBLAS K-part), D [64 tokens][16 experts] -> rpart, counted
        const int rp = (ru / 16) % pRS(P), eg = ru % 16, nch = P.rkn[rp] / 64, R = P.rR, rb = TPB2_RRB ? ru / (16 * pRS(P)) : 0;   // (rows 64 rb ..)
        __syncthreads();   // (TMEM address)
        tc_fence_after();
        const uint32_t rtm = static_cast<uint32_t>(T.misc[3]) + 3 * ACC_COLS;   // (accumulator buffer 3: unused until the role split)
        // x in sub-parts of rxch chunks (one sub-part = the whole part when it fits 96 KB), double-buffered at sb and sb + XB; the K16 chain
        // runs over the sub-parts in K order (one accumulator: == the single-shot chain)
        const int rxch = P.rxch, nsub = (nch + rxch - 1) / rxch;
        const uint32_t XB = static_cast<uint32_t>(rxch * R) * 128u;
        if (tid == 0) {
            for (int j = 0; j < nsub; ++j) {
                if (j >= 2) mbar_wait(bar((j & 1) ? B_RF1 : B_RF0), static_cast<uint32_t>((j - 2) >> 1) & 1u);   // (the MMAs of sub-part j - 2 done)
                mbar_expect_tx(bar((j & 1) ? B_RX1 : B_RX), XB);
                tma3d(sb + (j & 1) * XB, &P.maprx, 0, 64 * rb, P.rk0[rp] / 64 + j * rxch, bar((j & 1) ? B_RX1 : B_RX));
            }
        }
        if (warp == 1) {
            mbar_wait(bar(B_RW), 0);
            // kind::f16: D f32 (bit 4), A / B bf16, K-major, N = 16 (>> 3 at 17), M = 64 (R <= 64) or 128 (>> 4 at 24)
            const uint32_t idf = (1u << 4) | (1u << 7) | (1u << 10) | ((16u >> 3) << 17) | (((R > 64 ? 128u : 64u) >> 4) << 24);
            for (int j = 0; j < nsub; ++j) {
                mbar_wait(bar((j & 1) ? B_RX1 : B_RX), static_cast<uint32_t>(j >> 1) & 1u);
                tc_fence_after();
                if (elect_one()) {
                    const int cn = min(rxch, nch - j * rxch);
                    for (int c2 = 0; c2 < cn; ++c2) {
                        const int c = j * rxch + c2;
                        const uint64_t ad = desc_sw128(sb + (j & 1) * XB + static_cast<uint32_t>(c2 * R * 128)),
                                       bd = desc_sw128(sb + RW_OFF + static_cast<uint32_t>(c * 2048));
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %4, 0;\n tcgen05.mma." TPB2_CG ".kind::f16 [%0], %1, %2, %3, q;\n}"
                                         ::"r"(rtm), "l"(ad + 2 * k), "l"(bd + 2 * k), "r"(idf), "r"(static_cast<uint32_t>((c | k) != 0)) : "memory");
                    }
                    if (j + 2 < nsub) tc_commit(bar((j & 1) ? B_RF1 : B_RF0));
                    if (j == nsub - 1) tc_commit(bar(B_RM));
                }
                __syncwarp();
            }
        }
        if (warp >= 2 && warp <= 5) {   // TMEM lane quadrant q = warp % 4: D rows 16 q .. 16 q + 15 in its lanes 0..15 (M 64), rows 32 q + lane (M 128)
            const int q = warp & 3;
            mbar_wait(bar(B_RM), 0);
            tc_fence_after();
            float rv[16];
            tc_ld16_wait(rtm + (static_cast<uint32_t>(32 * q) << 16), rv);
            const int t = (R > 64 ? 32 : 16) * q + lane + 64 * rb;
            if ((R > 64 || lane < 16) && t < M) {
                float* dst = P.rpart + (static_cast<size_t>(rp) * M + t) * NEXP + 16 * eg;
#pragma unroll
                for (int j = 0; j < 16; j += 4) *reinterpret_cast<float4*>(dst + j) = make_float4(rv[j], rv[j + 1], rv[j + 2], rv[j + 3]);
            }
            tc_fence_before();
            named_bar(5, 128);
            if (warp == 2 && lane == 0) atom_add_acq_rel(P.tcnt + par * MAXM * NRT + 300, 1u);
        }
        if (warp == 1 && lane == 0) xt.rec(XT_PT(78));
    }
    if (rcta && tid == 0) xt.rec(XT_END(76));
    if (rfc) {   // ---- dsv3 router expert rfe: per thread 4 K iterations x 8 FMAs per token (dsv3's order), warp xor-sum 16 .. 1, then
                 // 0 + the group's 4 warp sums in warp order (T.am: dead until the FC1 epilogue)
        if (lane == 0 && (warp & 3) == 0) xt.rec(XT_BEG(76));
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const uint4* xr = reinterpret_cast<const uint4*>(P.xb) + (tid & 127);
#pragma unroll
        for (int i = 0; i < DIM / 1024; ++i) {
            uint4 xv[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) xv[m] = m < M ? __ldcg(xr + m * (DIM / 8) + 128 * i) : make_uint4(0u, 0u, 0u, 0u);
            const uint32_t wv[4] = {rwv[i].x, rwv[i].y, rwv[i].z, rwv[i].w};
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint32_t a[4] = {xv[m].x, xv[m].y, xv[m].z, xv[m].w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[m] = __fmaf_rn(bf16lo(a[j]), bf16lo(wv[j]), acc[m]);
                    acc[m] = __fmaf_rn(bf16hi(a[j]), bf16hi(wv[j]), acc[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) acc[m] = acc[m] + __shfl_xor_sync(0xffffffffu, acc[m], o);
        float* rsm = T.am + (warp >> 2) * 16;   // [token][warp of the group]
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < 4; ++m) if (m < M) rsm[m * 4 + (warp & 3)] = acc[m];
        }
        named_bar(12 + (warp >> 2), 128);
        if ((warp & 3) == 0) {
            if (lane < M) {
                float fs = 0.f;
#pragma unroll
                for (int w = 0; w < 4; ++w) fs += rsm[lane * 4 + w];
                P.rpart[lane * NEXP + rfe] = fs;
            }
            __syncwarp();
            if (lane == 0) { atom_add_acq_rel(P.tcnt + par * MAXM * NRT + 300, 1u); xt.rec(XT_END(76)); }
        }
    }
    if ((pRS(P) > 0 || pRF(P)) && tid == 0) xt.rec(XT_BEG(77));
    if (pRS(P) > 0 || pRF(P)) {   // every router unit's K-part logits in (acquire), before the gate reads them
        if (tid == 0) wait_ge(P.tcnt + par * MAXM * NRT + 300, static_cast<unsigned>(pRF(P) ? NEXP : 16 * pRS(P) * (TPB2_RRB ? P.rB : 1)));
        __syncthreads();
        if (tid == 0) xt.rec(XT_END(77));
    }
    // ---- routing (fused gate) + MXFP8 quantization of x
    const bool small = pGL(P) && M <= TPB2_SMALLM;
    int* gids = reinterpret_cast<int*>(smem + PRO_GIDS);
    if (lane == 0) xt.rec(XT_BEG(3));
    if (pGL(P)) {
        if (small && TPB2_PGATE1 && M == 1) {   // (the one token by warps 0..7 together)
            if (warp < 8) {
                int id = 0; float w = 0.f;
                gate_token_cta(P, P.glog, pNP(P), warp, lane, reinterpret_cast<uint64_t*>(smem + PRO_IDS),
                               reinterpret_cast<float*>(smem + PRO_IDS + 8 * GATE_TOPK * 8), id, w);
                if (warp == 0 && lane < pTK(P)) {
                    gids[lane] = id; T.srw[lane] = w; T.sids[lane] = static_cast<int16_t>(id);
                    if (b == 0) { P.ids[lane] = id; P.rw[lane] = w; }
                }
            }
        } else if (small) {   // every CTA gates every token (warp w: tokens w, w + 8, ..); CTA 0 also publishes them
            for (int t = warp; t < M; t += THREADS / 32) {
                int id; float w;
                if (TPB2_GWARM) { const GateR r = gate_call(P, P.glog + static_cast<size_t>(t) * NEXP, pNP(P), lane, &xt); id = r.id; w = r.wt; }
                else gate_token(P, P.glog + static_cast<size_t>(t) * NEXP, pNP(P), lane, id, w, &xt);
                if (lane < pTK(P)) {
                    gids[t * pTK(P) + lane] = id; T.srw[t * pTK(P) + lane] = w; T.sids[t * pTK(P) + lane] = static_cast<int16_t>(id);
                    if (b == 0) { P.ids[t * pTK(P) + lane] = id; P.rw[t * pTK(P) + lane] = w; }
                }
            }
        } else {   // token t by CTA t % G (warp (t / G) % 8), counted on gcnt[par]
            unsigned* c = P.gcnt + par;
            if (TPB2_PGATE) {   // (warps 0..7 together per token; warps 8..11 go on to the x quantization)
                uint64_t* cand = reinterpret_cast<uint64_t*>(smem + PRO_IDS);
                float* sca = reinterpret_cast<float*>(smem + PRO_IDS + 8 * GATE_TOPK * 8);
                if (warp < 8)
                    for (int t = b; t < M; t += G) {
                        int id = 0; float w = 0.f;
                        gate_token_cta(P, P.glog + static_cast<size_t>(t) * NEXP, pNP(P), warp, lane, cand, sca, id, w);
                        if (warp == 0) {
                            if (lane < pTK(P)) { P.ids[t * pTK(P) + lane] = id; P.rw[t * pTK(P) + lane] = w; }
                            __syncwarp();
                            if (lane == 0) { fence_acq_rel(); atomicAdd(c, 1u); }
                        }
                        named_bar(6, 256);   // (cand / sca reused by the next token)
                    }
            } else
            for (int t = b + warp * G; t < M; t += (THREADS / 32) * G) {
                int id; float w;
                if (TPB2_GWARM) { const GateR r = gate_call(P, P.glog + static_cast<size_t>(t) * NEXP, pNP(P), lane, &xt); id = r.id; w = r.wt; }
                else gate_token(P, P.glog + static_cast<size_t>(t) * NEXP, pNP(P), lane, id, w, &xt);
                if (lane < pTK(P)) { P.ids[t * pTK(P) + lane] = id; P.rw[t * pTK(P) + lane] = w; }
                __syncwarp();
                if (lane == 0) { fence_acq_rel(); atomicAdd(c, 1u); }
            }
        }
    }
    if (lane == 0) { xt.rec(XT_END(3)); xt.rec(XT_BEG(4)); }
    IdPre idp;   // (IDV 2) the routing ids: loads issued now, in flight over the x quantization
    idp.on = 0;
    if (TPB2_IDV == 2 && !pGL(P)) idv_load(P, idp, tid);
    // TPB2_GATHER: x -> MXFP8 (+ shared per-128) by token, independent of the routing: right after (and on other warps than) the gate
    if (TPB2_GATHER) {
        if (TPB2_QW811 && TPB2_PGATE && !small && pGL(P)) {   // (PGATE keeps warps 0..7 busy: warps 8..11 quantize every unit, <= 4 each,
            if (warp >= 8)                                   // their loads in one round trip)
                quant_x(P, nullptr, P.xflag + par * MAXM, lane, (THREADS / 32 - 1 - warp) * G + b, 4 * G, false);
        } else {
            quant_x(P, nullptr, P.xflag + par * MAXM, lane, (THREADS / 32 - 1 - warp) * G + b, (THREADS / 32) * G, false);
        }
    }
    if (DBGP && tid == 0) DBGP[b * 32 + 2] = gtime();
    if (lane == 0) xt.rec(XT_END(4));
    if (pGL(P) && !small) {
        if (tid == 0) wait_ge(P.gcnt + par, static_cast<unsigned>(M));
        __syncthreads();
    }
    __syncthreads();
    if (DBGP && tid == 0) DBGP[b * 32 + 27] = gtime();
    if (TPB2_PF1 && small && warp == 7) {   // first FC1 work of this CTA from the ids alone (the tables' panel order: experts ascending)
        const int MT = M * pTK(P);
        uint32_t mw = 0;   // lane w < 8: present experts 32 w .. 32 w + 31
        if (lane < 8)
            for (int i = 0; i < MT; ++i) { const int e = gids[i]; if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP) && (e >> 5) == lane) mw |= 1u << (e & 31); }
        uint32_t wv[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) wv[w] = __shfl_sync(0xffffffffu, mw, w);
        if (lane == 0) {
            int Pt = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) Pt += __popc(wv[w]);
            const int Ps = (M + NMAX - 1) / NMAX, nshq1 = Ps * NT1S * 4, nwork1 = nshq1 + Pt * NT1;
            const int R1p = Pt * NT1 * NSTG1, totc1 = R1p + TPB2_SHCOST * Ps * NT1S * NSTG1S;
            const bool rr1 = TPB2_RR && Pt * NT1 * 4 >= TPB2_RR_MIN4 * G && nshq1 <= G;
            int item = -1, s0 = 0, sk = -1, sc0 = 0;   // routed item / first stage, or shared tile k / first chunk
            if (rr1) {
                if (b < nshq1) { sk = (b >> 2) % NT1S; sc0 = (b & 3) * (NSTG1S / 4); }
                else if (b < nwork1) item = b - nshq1;
            } else {
                const int x0 = fc1_lo(totc1, b, G), g0 = x0 < R1p ? x0 : R1p + (x0 - R1p) / TPB2_SHCOST;
                if (g0 < R1p) { item = g0 / NSTG1; s0 = g0 % NSTG1; }
                else if (g0 < R1p + Ps * NT1S * NSTG1S) { sk = ((g0 - R1p) / NSTG1S) % NT1S; sc0 = (g0 - R1p) % NSTG1S; }
            }
            if (item >= 0 && (TPB2_PF1 & 1)) {
                int pr = item / NT1, e = -1;   // the pr-th present expert
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    const int c = __popc(wv[w]);
                    if (e < 0 && pr < c) { uint32_t m2 = wv[w]; for (int i = 0; i < pr; ++i) m2 &= m2 - 1u; e = 32 * w + __ffs(m2) - 1; }
                    else if (e < 0) pr -= c;
                }
                const int tile = item % NT1;
                if (e >= 0)
                    for (int s = s0; s < min(NSTG1, s0 + 2); ++s) {
                        pf_tma3d(&P.map1, 0, e * (2 * INTER) + tile * 128, KC * s);
                        pf_bulk(P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * s) * 512, KC * 512);
                    }
            } else if (sk >= 0 && (TPB2_PF1 & 2) && pSHW(P)) {
                for (int c = sc0; c < min(NSTG1S, sc0 + 2); ++c) { pf_tma3d(&P.map1s, 0, 128 * sk, c); pf_tma3d(&P.map1s, 0, INTER + 128 * sk, c); }
            }
        }
        __syncwarp();
    }
    if (TPB2_RPRE && small && warp == 7 && lane == 0) {   // ---- RPRE: the first routed piece's first stages, from the gate ids
        static_assert(!TPB2_RPRE || (TPB2_SHEARLY && TPB2_RPRE <= TPB2_SHPRE && TPB2_RPRE <= NS && TPB2_RRSPLIT == 1 && TPB2_SHW1 && TPB2_PUT1
                                     && !TPB2_PAIR && !TPB2_CSPLIT && TPB2_RR && TPB2_RR_MIN4 == 0), "RPRE: stages 0.. free in the prologue, rr PUT1 order");
        int npr = 0;
        const int r = pb - nshq0;   // (rr, PUT1, SHW1 waves: unit pb >= nshq starts with routed item pb - nshq, stages 0 ..)
        if (r >= 0) {
            const int MT = M * pTK(P);
            uint32_t wv[8];
#pragma unroll
            for (int w = 0; w < 8; ++w) wv[w] = 0u;
            for (int i = 0; i < MT; ++i) {
                const int e = gids[i];
                if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
#pragma unroll
                    for (int w = 0; w < 8; ++w) if ((e >> 5) == w) wv[w] |= 1u << (e & 31);
                }
            }
            int Pt = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) Pt += __popc(wv[w]);
            if (r < Pt * NT1U) {
                int pr = r / NT1U, e = -1;   // the pr-th present expert (panel pr)
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    const int c = __popc(wv[w]);
                    if (e < 0 && pr < c) { uint32_t m2 = wv[w]; for (int i = 0; i < pr; ++i) m2 &= m2 - 1u; e = 32 * w + __ffs(m2) - 1; }
                    else if (e < 0) pr -= c;
                }
                const int tile = r % NT1U;
                npr = TPB2_RPRE;
                for (int j = 0; j < npr; ++j) {
                    mbar_expect_tx(bar(B_FULL + j), KC * 8192 + KC * 512);
                    tma3d(sb + OFF_A + j * A_BYTES, &P.map1, 0, e * (2 * INTER) + tile * 128, KC * j, bar(B_FULL + j), 0x12F0000000000000ull);
                    bulk_g2s(sb + OFF_SFA + j * KC * 512, P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * j) * 512, KC * 512, bar(B_FULL + j));
                }
            }
        }
        T.misc[30] = npr;   // (the producer skips these fills; read after the role split)
    }
    unsigned xv[16];   // (warp 6, TPB2_GATHER) every token's x flag, acquire-loaded here: in flight over the tables, checked after the role split
    if (TPB2_GATHER && warp == 6) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
            xv[i] = lane + 32 * i < M ? (TPB2_XRLX ? ld_relaxed(P.xflag + par * MAXM + lane + 32 * i) : ld_acquire(P.xflag + par * MAXM + lane + 32 * i)) : 8u;
        if (XT_ON && (XT_MASK & 1u)) {   // (trace, prologue group: when the flags are in -- forces their completion here)
            unsigned sx = 0u;
#pragma unroll
            for (int i = 0; i < 16; ++i) sx += xv[i];
            if (lane == 0) xt.rec(XT_PT(8), sx);
        }
    }
    if (tid == 0) xt.rec(XT_BEG(5));
    if (TPB2_TABW && TPB2_GATHER && small && M <= TPB2_TABW_MAXM) { if (warp == 0) build_tables_warp(P, smem, T, gids, lane); __syncthreads(); }
    else build_tables(P, smem, T, small ? gids : nullptr, &xt, [&]() {
        // TPB2_PFF: this CTA's first routed FC1 piece's first stages (weights + SFA) -> L2 while the tables / scheduling finish (the
        // exact bytes its first TMA loads; DRAM is idle in the prologue)
        if (!TPB2_PFF || PR || CS || tid != 96) return;
        const int Ptot_ = T.misc[2], Psh_ = pSHW(P) ? (M + NMAX - 1) / NMAX : 0, nshq_ = Psh_ * NT1SU * SHQ;
        if (!TPB2_RR || TPB2_RR_MIN4 != 0 || nshq_ > PG) return;   // (rr every M only)
        const int rS_ = (TPB2_LASTM && TPB2_RRSPLIT >= 2) ? ((TPB2_RRSPLIT >= 4 && nshq_ + Ptot_ * NT1U * 4 <= PG) ? 4 : (nshq_ + Ptot_ * NT1U * 2 <= PG ? 2 : 1)) : 1;
        const int swv_ = (SHQ == 1 && TPB2_SHW1) ? TPB2_SHW1_W : 0;
        const int r = pb >= nshq_ ? pb - nshq_ : (swv_ ? swv_ * (PG - nshq_) + pb : pb + PG - nshq_);
        if (r >= Ptot_ * NT1U * rS_) return;
        const int item = r / rS_, s0 = (r % rS_) * (NSTG1 / rS_), s1 = s0 + min(NSTG1 / rS_, TPB2_PFF);
        int e, n, eb;
        panel_of(T, item / NT1U, e, n, eb);
        const int tile = item % NT1U;
        for (int s = s0; s < s1; ++s) {
            pf_tma3d(&P.map1, 0, e * (2 * INTER) + tile * 128, KC * s);
            pf_bulk(P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * s) * 512, KC * 512);
        }
    }, &idp);
    if (DBGP && tid == 0) DBGP[b * 32 + 1] = gtime();
    if (tid == 0) xt.rec(XT_END(5));
    if (M <= TPB2_L2W2_MAXM && warp == 0 && P.wp[1]) {   // every panel's FC2 weights (+ scales) -> L2: certain traffic, its first-load latency
        const int Pt = T.misc[2], per = 17;                 // off the FC2 critical path (16 x 64 KB of w2 + 64 KB of w2_sf per expert)
        for (int c = b + G * lane; c < Pt * per; c += 32 * G) {
            int e, n, eb;
            panel_of(T, c / per, e, n, eb);
            const int k = c % per;
            pf_bulk(k < 16 ? P.wp[1] + (static_cast<size_t>(e) * 16 + k) * 65536 : P.w2_sf + static_cast<size_t>(e) * 65536, 65536);
        }
    }
    if (M <= TPB2_L2ALL_MAXM && warp == 0 && P.wp[0]) {   // all needed weights -> L2 (64 KB pieces over the grid's lanes)
        const int Pt = T.misc[2], per = 51, nch = Pt * per + (pSHW(P) ? 96 : 0);
        for (int c = b + G * lane; c < nch; c += 32 * G) {
            const uint8_t* src;
            if (c < Pt * per) {
                int e, n, eb;
                panel_of(T, c / per, e, n, eb);
                const int k = c % per;
                src = k < 32 ? P.wp[0] + (static_cast<size_t>(e) * 32 + k) * 65536
                    : k < 48 ? P.wp[1] + (static_cast<size_t>(e) * 16 + (k - 32)) * 65536
                    : k < 50 ? P.w13_sf + (static_cast<size_t>(e) * 2 + (k - 48)) * 65536
                             : P.w2_sf + static_cast<size_t>(e) * 65536;
            } else {
                const int k = c - Pt * per;
                src = k < 64 ? P.wp[2] + static_cast<size_t>(k) * 65536 : P.wp[3] + static_cast<size_t>(k - 64) * 65536;
            }
            pf_bulk(src, 65536);
        }
    }
    if (!TPB2_GATHER) {
        quant_x(P, T.tke, P.xflag + par * MAXM, lane, warp * G + b, (THREADS / 32) * G, true);   // x -> MXFP8 (+ shared per-128), into its panels' images
        if (DBGP && tid == 0) DBGP[b * 32 + 2] = gtime();
    }
    const int Ptot = T.misc[2];
    const int Psh = pSHW(P) ? (M + NMAX - 1) / NMAX : 0;   // shared panels (FC1 items / FC2 units after the routed ones)
    if (tid == 33) { int* q = reinterpret_cast<int*>(smem + OFF_TAB + TAB_T1Q); q[T1Q] = 0; q[T1Q + 1] = 0; q[T1Q + 2] = 0; T.misc[26] = 0; }
    if (tid == 0) T.misc[4] = Psh;
    // FC1 stream-K over cost units: a routed stage (16 KB of e2m1) = 1, a shared stage (2 x 16 KB of e4m3) = 2.  Unified stage index:
    // routed stages [0, R1), shared [R1, R1 + S1); CTA b's range = the stages of its cost range [totc b / G, totc (b + 1) / G)
    // (all scheduling in units: pb of PG; PAIR unit = 2 tiles / row tiles of the CTA pair)
    const int nshq = Psh * NT1SU * SHQ * (shgu ? 2 : 1);
    const int nworkc = nshq / 2 + Ptot * NT1;   // CSPLIT unit entries
    // (CSPLIT: cluster round robin -- shared pieces on CTAs 0 .. nshq - 1 = units 0 .. nshq / 2 - 1 -- only when every unit entry fits one
    // wave (M = 1): paired A/B M1 -7.8%, more waves +4..13%)
    const bool csr = CS && nshq <= G && nworkc <= PG1;
    // (RR_MIN4 0, no CSPLIT / PAIR: rr at every M -- the host checks nshq <= G -- so the stream-K scheduling and its owner / contributor
    // merges compile away: less cold code on the critical paths)
    constexpr bool RR_ALWAYS = TPB2_RR && TPB2_RR_MIN4 == 0 && !TPB2_CSPLIT && !TPB2_PAIR;
    const bool rr = RR_ALWAYS || (!csr && TPB2_RR && (Ptot * NT1U * 4 >= TPB2_RR_MIN4 * PG || Ptot * NT1U * 4 < TPB2_RR_LO4 * PG) && nshq <= PG);
    constexpr bool WHOLE = TPB2_WHOLE && RR_ALWAYS && TPB2_RRSPLIT == 1 && !TPB2_RRTAIL2 && TPB2_SHW1 == 1;   // (no split FC1 items)
    static_assert(!TPB2_WHOLE || WHOLE, "TPB2_WHOLE: rr always, RRSPLIT 1, no RRTAIL2, SHW1 1");
    const int rS = (rr && TPB2_LASTM && TPB2_RRSPLIT >= 2) ? ((TPB2_RRSPLIT >= 4 && nshq + Ptot * NT1U * 4 <= PG) ? 4 : (nshq + Ptot * NT1U * 2 <= PG ? 2 : 1)) : 1;
    const bool skq = shearly && !rr && !csr;   // stream-K with the shared pieces on units 0 .. nshq - 1, routed stages over the rest
    const int R1 = Ptot * NT1U * NSTG1, S1 = Psh * NT1SU * NSTG1S, totc = skq ? R1 : R1 + TPB2_SHCOST * S1;
    auto ustart = [&](int c) {
        if (skq) return c <= nshq ? 0 : fc1_lo(R1, c - nshq, PG - nshq);
        const int x = fc1_lo(totc, c, PG); return x < R1 ? x : R1 + (x - R1) / TPB2_SHCOST;
    };
    // FC2 unit width: the widest of 4 / 2 / 1 row tiles (PAIR: row-tile pairs) that still gives every unit >= 2 FC2 units
    const int Pall = Ptot + Psh;
    const int fc2g0 = (TPB2_FC2G8 && Pall * (NRT2U / 8) >= 2 * PG) ? 8 : (Pall * (NRT2U / 4) >= 2 * PG ? 4 : (Pall * (NRT2U / 2) >= 2 * PG ? 2 : 1));
    const int fc2g = min(fc2g0, M >= TPB2_FC2G4_MINM ? 4 : TPB2_FC2GMAX), nrg = NRT2U / fc2g;   // (v60: 4-row-tile units cap at M >= 64)
    static_assert(!(TPB2_FC2VAR && TPB2_T1W), "FC2VAR: RTPUB queue only");
    // FC2 descriptor (kinds 2 / 4): panel, first unit row tile (before trow), row tiles.  FC2VAR: d.y = panel NRT2U + row tile, d.z = count
    auto f2p = [&](const int4& d) -> int { return TPB2_FC2VAR ? d.y / NRT2U : d.y / nrg; };
    auto f2r0 = [&](const int4& d) -> int { return TPB2_FC2VAR ? d.y % NRT2U : (d.y % nrg) * fc2g; };
    auto f2n = [&](const int4& d) -> int { return TPB2_FC2VAR ? d.z : fc2g; };
    const int lo = rr ? 0 : ustart(pb), hi = rr ? 0 : ustart(pb + 1);
    // round-robin FC1 (rr): work list j = shared quarter pieces (Psh x 4 items x 4 quarters of 8 chunks: CTA j, the owner of an item is
    // its quarter 0, the contributors the next 3 CTAs -- every CTA's only shared piece is its first work) then routed items (whole, no
    // split), entry j on CTA j % G in increasing j
    const int nwork = nshq + Ptot * NT1U * rS;   // (rS > 1: routed entries = item K-pieces)
    constexpr int tS = 1;
    // TPB2_RRTAIL2: (rS == 1) the last nwork % PG entries (routed items Ptot NT1U - tr ..) as tS2 K-pieces on units 0 .. tr tS2 - 1
    const int tr0 = (rr && TPB2_LASTM && TPB2_RRTAIL2 && rS == 1) ? nwork % PG : 0;
    const int tS2 = tr0 == 0 || tr0 > Ptot * NT1U ? 1 : (4 * tr0 <= PG ? 4 : (2 * tr0 <= PG ? 2 : 1));
    const int tr = tS2 > 1 ? tr0 : 0, nwfull = nwork - tr, ritem0 = Ptot * NT1U - tr;   // (tail items ritem0 ..)
    // tokens of a work descriptor's panel (kinds: 1 routed FC1 piece, 2 routed FC2 unit, 3 shared FC1 piece, 4 shared FC2 unit)
    auto panel_n = [&](const int4& d) -> int {
        int e, n, eb;
        if (d.x == 1) { panel_of(T, d.y / NT1U, e, n, eb); return n; }
        if (d.x == 2) { panel_of(T, f2p(d), e, n, eb); return n; }
        if (d.x == 3) return min(NMAX, M - NMAX * (d.y / NT1SU));
        return min(NMAX, M - NMAX * (f2p(d) - Ptot));
    };
    if (TPB2_DLY_SITE == 1) dly_spin(TPB2_DLY_NS);
    if (lane == 0) xt.rec(XT_PT(69));
    fence_proxy_async_smem();   // prologue scratch (generic) precedes the ring's async-proxy writes
    __syncthreads();
    if (lane == 0) xt.rec(XT_PT(70));
    tc_fence_after();
    const uint32_t tmem = static_cast<uint32_t>(T.misc[3]);
    int4* ring = T.ring;   // (written before the slot's WF arrive, read after its wait: the mbarrier orders it)

    if (warp == 0) {
        // ======================================================== producer: work descriptors (the scheduler: 1-CTA, or the PAIR leader for both
        // CTAs' rings) + this CTA's own weight tile stages (A) + SFA.  Descriptors are unit-level (PAIR: unit = 2 tiles; tile = 2 u + crank)
        if (lane == 0) {
            int fills = 0, w = 0;
            const int nrpre = (TPB2_RPRE && small) ? T.misc[30] : 0;   // (RPRE: ring stages already issued)
            unsigned long long pw1 = 0, pw2 = 0;   // (dbg) producer waits for free stages, FC1 / FC2
            bool in_fc2 = false;
            const uint32_t ring_peer = PR ? peer_addr(su32(ring), crank ^ 1u) : 0u, wf_peer = PR ? peer_addr(bar(B_WF), crank ^ 1u) : 0u;
            const uint32_t full_lead = PR ? peer_addr(bar(B_FULL), 0u) : 0u;   // (PAIR peer: the leader's FULL barriers)
            auto push = [&](int4 d) {   // (scheduler) -> this CTA's ring and (PAIR) the peer's
                const int slot = w % WR;
                if (w >= WR) mbar_wait(bar(B_WE + slot), ((w / WR) - 1) & 1);
                ring[slot] = d;
                if (PR) { st_cluster_v4(ring_peer + 16 * slot, d); mbar_arrive_remote(wf_peer + 8 * slot); }
                mbar_arrive(bar(B_WF + slot));
                ++w;
                if (d.x >= 0) xt.rec(XT_BEG(10), xd0(d), xd1(d)); else xt.rec(XT_PT(13));
            };
            auto stage = [&](const CUtensorMap* map, int row, int ch0, const uint8_t* sfa, uint32_t atx) {   // KC chunks of one 128-row tile
                if (TPB2_RPRE && fills < nrpre) { ++fills; return; }   // (RPRE: issued from the gate ids before the tables)
                const int st = fills % NS;
                const unsigned long long tw = DBGP ? ctime() : 0ull;
                if (fills >= NS) { xt.rec(XT_BEG(11)); mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1); xt.rec(XT_END(11)); }
                if (DBGP) { if (in_fc2) pw2 += ctime() - tw; else pw1 += ctime() - tw; if (fills == 0) DBGP[b * 32 + 28] = gtime(); }
                // (PAIR: both CTAs' A bytes complete on the leader's FULL -- the leader expects 2 x; the peer's local FULL: its SFA (+ B / SFB))
                mbar_expect_tx(bar(B_FULL + st), (PR ? (lead ? 2 * atx : 0u) : atx) + ((TPB2_PROBE & 8) ? 0 : KC * 512));
                if (PR && !lead) tma3d_cg2(sb + OFF_A + st * A_BYTES, map, 0, row, ch0, full_lead + 8 * st);
                else tma3d(sb + OFF_A + st * A_BYTES, map, 0, row, ch0, bar(B_FULL + st), (TPB2_SHPOL && map == &P.map2s) ? SH_HINT : 0x12F0000000000000ull);
                if (!(TPB2_PROBE & 8)) bulk_g2s(sb + OFF_SFA + st * KC * 512, sfa, KC * 512, bar(B_FULL + st));
                ++fills;
            };
            auto stage_s1 = [&](int k, int c) {   // shared FC1: chunk c of the gate tile (rows 128 k ..) and the up tile (rows INTER + 128 k ..)
                if (fills < npre) { ++fills; return; }   // (issued before the PDL wait)
                const int st = fills % NS;
                if (fills >= NS) { xt.rec(XT_BEG(11)); mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1); xt.rec(XT_END(11)); }
                if (shh >= 0) {   // (SHGU: stage c = chunks 2 c, 2 c + 1 of this CTA's gate or up rows + their scales)
                    mbar_expect_tx(bar(B_FULL + st), 2 * 16384 + 2 * 512);
                    for (int h2 = 0; h2 < 2; ++h2) {
                        tma3d(sb + OFF_A + st * A_BYTES + h2 * 16384, &P.map1s, 0, shh * INTER + 128 * k, 2 * c + h2, bar(B_FULL + st), SH_HINT);
                        bulk_g2s(sb + OFF_SFA + st * KC * 512 + h2 * 512, P.sfa1s + (static_cast<size_t>(shh * (INTER / 128) + k) * NCH1 + 2 * c + h2) * 512, 512, bar(B_FULL + st));
                    }
                    ++fills;
                    return;
                }
                mbar_expect_tx(bar(B_FULL + st), (PR ? (lead ? 4 * 16384 : 0) : 2 * 16384) + 2 * 512);
                if (PR && !lead) {
                    tma3d_cg2(sb + OFF_A + st * A_BYTES, &P.map1s, 0, 128 * k, c, full_lead + 8 * st);
                    tma3d_cg2(sb + OFF_A + st * A_BYTES + 16384, &P.map1s, 0, INTER + 128 * k, c, full_lead + 8 * st);
                } else {
                    tma3d(sb + OFF_A + st * A_BYTES, &P.map1s, 0, 128 * k, c, bar(B_FULL + st), SH_HINT);
                    tma3d(sb + OFF_A + st * A_BYTES + 16384, &P.map1s, 0, INTER + 128 * k, c, bar(B_FULL + st), SH_HINT);
                }
                bulk_g2s(sb + OFF_SFA + st * KC * 512, P.sfa1s + (static_cast<size_t>(k) * NCH1 + c) * 512, 512, bar(B_FULL + st));
                bulk_g2s(sb + OFF_SFA + st * KC * 512 + 512, P.sfa1s + (static_cast<size_t>(INTER / 128 + k) * NCH1 + c) * 512, 512, bar(B_FULL + st));
                ++fills;
            };
            auto issue = [&](const int4& d) {   // this CTA's stages of a descriptor
                if (d.x == 1) {
                    const int item = titem(d.y), tile = item % NT1;
                    int e, n, eb;
                    panel_of(T, item / NT1, e, n, eb);
                    for (int s = d.z; s < d.w; ++s)
                        stage(&P.map1, e * (2 * INTER) + tile * 128, KC * s,
                              P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * s) * 512, KC * 8192);
                } else if (d.x == 3) {
                    const int k = titemS(d.y) % NT1S;
                    for (int s = d.z; s < d.w; ++s) stage_s1(k, s);
                } else {
                    const int p = f2p(d), r0 = f2r0(d), cn = f2n(d);
                    if (p < Ptot) {
                        int e, n, eb;
                        panel_of(T, p, e, n, eb);
                        for (int j = 0; j < cn; ++j) {
                            const int r = trow(r0 + j);
                            for (int s = 0; s < NSTG2; ++s)
                                stage(&P.map2, e * DIM + r * 128, KC * s, P.w2_sf + ((static_cast<size_t>(e) * NRT2 + r) * NCH2 + KC * s) * 512, KC * 8192);
                        }
                    } else {
                        for (int j = 0; j < cn; ++j) {
                            const int r = trow(r0 + j);
                            for (int s = 0; s < NSTG2; ++s)
                                stage(&P.map2s, r * 128, KC * s, P.sfa2s + (static_cast<size_t>(r) * NCH2 + KC * s) * 512, KC * 16384);
                        }
                    }
                }
                xt.rec(XT_END(10));
            };
            if (lead) {
                int nf1 = 0;   // (EPIFIRST: FC1 descriptors pushed)
                auto put = [&](int4 d) { push(d); issue(d); ++nf1; };
                const int swv = (rr && SHQ == 1 && TPB2_SHW1) ? TPB2_SHW1_W : 0;   // (SHW1: units < nshq hold a whole shared item for swv waves)
                auto put_r = [&](int r) { const int kp = r % rS, rs0 = kp * (NSTG1 / rS); put(make_int4(1, r / rS, rs0, rs0 + NSTG1 / rS)); };
                if (rr && swv) {   // shared item pb first; routed entry r: waves < swv over units nshq .., then every unit round robin
                    if (pb < nshq) put(make_int4(3, shgu ? pb >> 1 : pb, 0, shgu ? NSTG1S / 2 : NSTG1S));
                    const int nr = Ptot * NT1U * rS, gs = PG - nshq, head = swv * gs;
                    if (TPB2_PUT1) {   // (one put site: waves w2 < swv (units >= nshq), then round robin from head)
                        const int nw0 = pb >= nshq ? swv : 0;
                        for (int k = 0;; ++k) {
                            const int r = k < nw0 ? k * gs + (pb - nshq) : head + pb + (k - nw0) * PG;
                            if (r >= nr) { if (k < nw0) continue; break; }
                            put_r(r);
                        }
                    } else {
                    if (pb >= nshq) for (int w2 = 0; w2 < swv; ++w2) { const int r = w2 * gs + (pb - nshq); if (r < nr) put_r(r); }
                    for (int r = head + pb; r < nr; r += PG) put_r(r);
                    }
                } else if (rr) {
                    for (int j = pb; j < nwfull; j += PG) {
                        if (j < nshq) {   // shared piece: unit item j / SHQ, chunks SHC (j % SHQ) ..
                            const int s0 = (j % SHQ) * SHC;
                            put(make_int4(3, j / SHQ, s0, s0 + SHC));
                        } else {          // routed item (unit) in rS K-pieces
                            const int kp = (j - nshq) % rS, rs0 = kp * (NSTG1 / rS);
                            put(make_int4(1, (j - nshq) / rS, rs0, rs0 + NSTG1 / rS));
                        }
                    }
                    if (pb < tr * tS2) {   // tail piece: item ritem0 + pb / tS2, K-piece pb % tS2
                        const int k = pb % tS2, ts0 = k * (NSTG1 / tS2);
                        put(make_int4(1, ritem0 + pb / tS2, ts0, ts0 + NSTG1 / tS2));
                    }
                }
                if (csr) {   // CSPLIT: unit entries j = pb1, pb1 + PG1, ..: shared pieces 2 j + crank, then routed items (this rank's K-half)
                    for (int j = pb1; j < nworkc; j += PG1) {
                        if (j < nshq / 2) {
                            const int qg = 2 * j + static_cast<int>(crank), s0 = (qg % SHQ) * SHC;
                            put(make_int4(3, qg / SHQ, s0, s0 + SHC));
                        } else {
                            const int h0 = static_cast<int>(crank) * (NSTG1 / 2);
                            put(make_int4(1, j - nshq / 2, h0, h0 + NSTG1 / 2));
                        }
                    }
                }
                if (skq && pb < nshq) {   // this unit's shared piece (as in rr), then nothing routed (lo = hi)
                    const int s0 = (pb % SHQ) * SHC;
                    put(make_int4(3, pb / SHQ, s0, s0 + SHC));
                }
                for (int g = (rr || csr) ? hi : lo; g < hi;) {   // FC1 pieces (unified stage index over units: routed, then shared)
                    if (g < R1) {
                        const int item = g / NSTG1, s0 = g % NSTG1, s1 = min(NSTG1, hi - item * NSTG1);
                        put(make_int4(1, item, s0, s1));
                        g = item * NSTG1 + s1;
                    } else {
                        const int sg = g - R1, sitem = sg / NSTG1S, s0 = sg % NSTG1S, s1 = min(NSTG1S, hi - R1 - sitem * NSTG1S);
                        put(make_int4(3, sitem, s0, s1));
                        g = R1 + sitem * NSTG1S + s1;
                    }
                }
                if (DBGP) DBGP[b * 32 + 3] = gtime();
                xt.rec(XT_PT(14));
                if (TPB2_EPIFIRST && nf1 > 0) {   // the epilogue finishes this CTA's FC1 items (act + h publish) before FC2 traffic starts here
                    while (*reinterpret_cast<volatile int*>(T.misc + 26) < nf1) __nanosleep(64);
                }
                in_fc2 = true;
                // (FC2VAR: row tiles R in slot-major order; [0, RA) in units of fc2g, [RA, RB) pairs, [RB, RT) singles)
                const int RT = Pall * NRT2U;
                int RA = RT, RB = RT;
                if (TPB2_FC2VAR && fc2g >= 4) { RB = max(0, RT - 2 * ((PG + 1) / 2)); RA = RB >= 2 * PG ? ((RB - 2 * PG) / fc2g) * fc2g : 0; }
                else if (TPB2_FC2VAR && fc2g == 2 && fc2g0 >= 4) { RB = max(0, RT - 2 * ((PG + 1) / 2)); RA = RB; }   // (GMAX 2: pairs, then singles)
                // (RTMAJ: R >= L0 late, row-pair-major over the late slots; units there are pairs / singles)
                const int L0 = TPB2_RTMAJ ? (TPB2_RTMAJ >= 1000 ? (RA / NRT2U) * NRT2U : max(0, RT - TPB2_RTMAJ * NRT2U)) : RT;
                if (TPB2_RTMAJ && L0 < RA) { RA = L0; if (fc2g == 1) RB = L0; }
                const int nA = RA / fc2g, nB = (RB - RA) / 2;
                auto rmap = [&](int R, int& row) -> int {   // R -> slot, row tile
                    if (!TPB2_RTMAJ || R < L0) { row = R % NRT2U; return R / NRT2U; }
                    const int Rp = R - L0, nl = Pall - L0 / NRT2U, rt = Rp / (2 * nl), q = Rp - rt * 2 * nl;
                    row = 2 * rt + (q & 1);
                    return L0 / NRT2U + (q >> 1);
                };
                auto tk = [&](int u, int& R0) -> int {
                    if (u < nA) { R0 = u * fc2g; return fc2g; }
                    if (u < nA + nB) { R0 = RA + 2 * (u - nA); return 2; }
                    R0 = RB + (u - nA - nB); return 1;
                };
                auto tslot = [&](int u) -> int { if (!TPB2_FC2VAR) return u / nrg; int R0, row; tk(u, R0); return rmap(R0, row); };
                const int units = TPB2_FC2VAR ? nA + nB + (RT - RB) : Pall * nrg;
                // FC2 units in the order panels become ready: ticket u -> ready-list slot u / nrg (the panel whose h completed (u / nrg)-th), row
                // group u % nrg.  One ticket ahead, its slot loaded as soon as the ticket arrives (acquire: the panel's h)
                const unsigned* rl = P.rdy + par * (MAXP + 32);
                auto ldf = [&](const unsigned* a) -> unsigned { return TPB2_FC2RLX ? ld_relaxed(a) : ld_acquire(a); };
                if (fc2l) {   // FC2L: row tile pb of every panel (routed, then the shared one), weights streamed now
                    if (pb < NRT2U)
                        for (int p2 = 0; p2 < Pall; ++p2) { const int4 d = make_int4(p2 < Ptot ? 2 : 4, p2 * NRT2U + pb, 1, 0); push(d); issue(d); }
                }
                int tq = (fc2l || TPB2_PROBE_END == 1) ? units : static_cast<int>(atomicAdd(P.fc2_next + par, 1u));
                int tq2 = TPB2_FC2RLX ? static_cast<int>(atomicAdd(P.fc2_next + par, 1u)) : 0;   // (FC2RLX: the next ticket already in flight)
                unsigned pf = tq < units ? ldf(rl + tslot(tq)) : 0u;
                static_assert(!(TPB2_FC2CAP && TPB2_FC2RLX), "FC2CAP: one ticket at a time");
                const int fcap = (units + PG - 1) / PG + TPB2_FC2CAP - 1;   // (FC2CAP: tickets this unit may claim)
                int nclaim = 0;
                for (;;) {
                    const int u = tq;
                    if (u >= units) break;
                    const int rg = u % nrg;
                    if (f2s) {   // (static: slot = panel -- TPB2_FC2S_SHF: the shared panels first -- the B warp waits for the panel's h)
                        const int sl = tslot(u);
                        pf = static_cast<unsigned>((TPB2_FC2S_SHF ? (sl < Psh ? Ptot + sl : sl - Psh) : sl) + 1);
                    }
                    if (pf == 0u) {
                        xt.rec(XT_BEG(12), static_cast<uint32_t>(u));
                        const unsigned long long tw = DBGP ? ctime() : 0ull;
                        while ((pf = ldf(rl + tslot(u))) == 0u) {}
                        xt.rec(XT_END(12));
                        if (DBGP) DBGP[b * 32 + 21] += ctime() - tw;
                    }
                    const int p = static_cast<int>(pf) - 1;
                    int4 d = make_int4(p < Ptot ? 2 : 4, p * nrg + rg, 0, 0);
                    if (TPB2_FC2VAR) { int R0, row; const int cn = tk(u, R0); rmap(R0, row); d.y = p * NRT2U + row; d.z = cn; }
                    push(d);
                    if (TPB2_FC2RLX) {
                        const int tq3 = static_cast<int>(atomicAdd(P.fc2_next + par, 1u));   // (needed one iteration later)
                        pf = (tq2 < units && !f2s) ? ldf(rl + tslot(tq2)) : 0u;   // (in flight over issue(d))
                        tq = tq2; tq2 = tq3;
                    } else {
                        tq = (TPB2_FC2CAP && ++nclaim >= fcap) ? units : static_cast<int>(atomicAdd(P.fc2_next + par, 1u));
                        pf = (tq < units && !f2s) ? ld_acquire(rl + tslot(tq)) : 0u;
                    }
                    issue(d);
                }
                push(make_int4(-1, 0, 0, 0));
            } else {   // PAIR peer: this CTA's stages of the leader's descriptors (its ring, filled by the leader)
                for (int wr = 0;; ++wr) {
                    const int slot = wr % WR;
                    mbar_wait_cl(bar(B_WF + slot), (wr / WR) & 1);
                    const int4 d = ring[slot];
                    use4(d);
                    we_arrive(slot);
                    if (d.x < 0) break;
                    if (d.x == 2 || d.x == 4) in_fc2 = true;
                    xt.rec(XT_BEG(10), xd0(d), xd1(d));
                    issue(d);
                }
            }
            if (DBGP) { DBGP[b * 32 + 24] = pw1; DBGP[b * 32 + 20] = pw2; }
        }
        __syncwarp();
    } else if (warp == 7 && TPB2_SFCP_MMA) {
        // ======================================================== T1 worker (TPB2_T1W): pairs j = b + k G of (token t = j % M, 256-row tile
        // rt = j / M); each lane polls up to 2 of them, a completed pair (all 2 (valid slots + shared) FC2 row tiles counted) is T1'd by the warp
        if (TPB2_T1W) {
            const int npair = M * NRT, nmine = b < npair ? (npair - 1 - b) / G + 1 : 0;
            unsigned need[2] = {0u, 0u};
            const unsigned* cnt = P.tcnt + par * MAXM * NRT;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int k = lane + 32 * h;
                if (k < nmine) {
                    const int j = b + k * G, t = j % M;
                    unsigned nv = 0u;
                    for (int s2 = 0; s2 < pTK(P); ++s2) nv += static_cast<unsigned>(__ldcg(P.ids + t * pTK(P) + s2)) < static_cast<unsigned>(NEXP) ? 1u : 0u;
                    need[h] = 2u * (nv + (pSHW(P) ? 1u : 0u));
                }
            }
            unsigned long long left = nmine >= 64 ? ~0ull : ((1ull << nmine) - 1ull);
            __nv_bfloat16* stg7 = reinterpret_cast<__nv_bfloat16*>(smem + OFF_TAB + TAB_T1S);
            volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
            int qh = 0;   // units consumed
            bool epi_done = false;
            unsigned* tcw = P.tcnt + par * MAXM * NRT;
            auto drain = [&]() {   // count every queued FC2 unit's (token, 256-row tile) contributions (gpu-scope release, off the epilogue)
                const int qn = __shfl_sync(0xffffffffu, q[T1Q], 0);   // (warp-uniform)
                if (qn == qh) return;
                __threadfence_block();   // (CTA acquire of the units' outp / shout stores, observed through the queue)
                if (TPB2_T1W_MODE != 2) fence_acq_rel();   // (then released at gpu scope by every lane before its reds)
                for (; qh < qn; ++qh) {
                    const int v = q[qh % T1Q], dx = v >> 24, dy = v & 0xFFFFFF, pp = dy / nrg;
                    int nn, ebb = 0, tp = 0;
                    if (dx == 2) { int ee; panel_of(T, pp, ee, nn, ebb); }
                    else { tp = pp - Ptot; nn = min(NMAX, M - NMAX * tp); }
                    const int r0 = (dy % nrg) * fc2g, rt0 = r0 >> 1, nrt = ((r0 + fc2g - 1) >> 1) - rt0 + 1, per_rt = fc2g >= 2 ? 2 : 1;
                    for (int x = lane; x < nn * nrt; x += 32) {
                        const int i = x / nrt, rt = rt0 + (x - i * nrt);
                        int t;
                        unsigned add;
                        if (dx == 4) { t = NMAX * tp + i; add = static_cast<unsigned>(per_rt); }
                        else { const unsigned ev = T.ent[ebb + i]; t = static_cast<int>(ev & 511u); add = static_cast<unsigned>(__popc(ev >> 9) * per_rt); }
                        red_relaxed_add(tcw + t * NRT + rt, add);
                    }
                }
                __syncwarp();
                if (lane == 0) q[T1Q + 2] = qh;
            };
            for (;;) {
                const int qn = __shfl_sync(0xffffffffu, q[T1Q], 0), dn = __shfl_sync(0xffffffffu, q[T1Q + 1], 0);
                if (epi_done && qn == qh) break;   // (this CTA's FC2 counted: the pairs still open go to the tail, all warps)
                epi_done = dn != 0;   // (read before the drain: a unit queued before the flag is drained in this or the next sweep)
                drain();
                if (!left || TPB2_T1W_MODE == 1) { __nanosleep(100); continue; }
                unsigned rdy = 0u;   // this lane's completed pairs (bit h)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int k = lane + 32 * h;
                    if (k < nmine && ((left >> k) & 1ull)) {
                        const int j = b + k * G;
                        if (ld_relaxed(cnt + (j % M) * NRT + j / M) >= need[h]) rdy |= 1u << h;
                    }
                }
                const unsigned b0 = __ballot_sync(0xffffffffu, rdy & 1u), b1 = __ballot_sync(0xffffffffu, (rdy >> 1) & 1u);
                const unsigned long long got = static_cast<unsigned long long>(b0) | (static_cast<unsigned long long>(b1) << 32);
                if (!got) { __nanosleep(TPB2_T1W_SLEEP); continue; }   // (back off: 148 x 32 spinning lanes load the L2 under the weight stream)
                fence_acq_rel();   // (acquire: the outp / shared rows of the completed pairs)
                for (unsigned long long gm = got; gm; gm &= gm - 1ull) {
                    const int k = __ffsll(static_cast<long long>(gm)) - 1, j = b + k * G;
                    t1_pair(P, T, small, j / M, j % M, epoch, stg7, lane);
                }
                left &= ~got;
            }
            if (lane == 0) { T.misc[12] = static_cast<int>(static_cast<unsigned>(left)); T.misc[13] = static_cast<int>(static_cast<unsigned>(left >> 32)); }
        }
        const bool t1ll = t1ll_on(P);
        const bool t1l = t1l_on(M) && !TPB2_T1W && TPB2_RTPUB && !t1ll && !fc2l_on(P);
        if (t1ll) {   // ---- T1LL: pairs j = b, b + G, .. of (token j % M, 256-row tile j / M), polled from the LL images
            __nv_bfloat16* stg7 = reinterpret_cast<__nv_bfloat16*>(smem + OFF_TAB + TAB_T1S);
            for (int j = b; j < NRT * M; j += G) t1ll_pair(P, T, j / M, j % M, epoch, stg7, lane, &xt);
        }
        if (t1l) {   // ---- T1L: the FC2 units queued by the epilogue -> (token, 256-row tile) counts; a pair this warp completes is T1'd here
            volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
            __nv_bfloat16* stg7 = reinterpret_cast<__nv_bfloat16*>(smem + OFF_TAB + TAB_T1S);
            unsigned* tc = P.tcnt + par * MAXM * NRT + 1024;
            int c = 0;
            for (;;) {
                const int n = __shfl_sync(0xffffffffu, q[T1Q], 0), dn = __shfl_sync(0xffffffffu, q[T1Q + 1], 0);
                if (c == n) {
                    if (dn && c == __shfl_sync(0xffffffffu, q[T1Q], 0)) break;
                    __nanosleep(64);
                    continue;
                }
                if (lane == 0) xt.rec(XT_BEG(59), static_cast<uint32_t>(n - c));
                __threadfence_block();   // (the queued units' outp / shout rows, observed through the queue)
                fence_acq_rel();         // (released at gpu scope before every lane's count)
                for (; c < n; ++c) {
                    const int qv = q[c % T1Q], gp = (qv & 0xFFFFFF) / NRT2U, r0 = (qv & 0xFFFFFF) % NRT2U, cn = qv >> 24;
                    int nn, ebb = 0, sp = -1;
                    if (gp < Ptot) { int ee; panel_of(T, gp, ee, nn, ebb); }
                    else { sp = gp - Ptot; nn = min(NMAX, M - NMAX * sp); }
                    const int rtA = trow(r0) >> 1, rtB = trow(r0 + cn - 1) >> 1, nrt = rtB - rtA + 1;   // (T1 tiles of the unit's row tiles)
                    for (int x0 = 0; x0 < nn * nrt; x0 += 32) {   // (rounds of 32 (token, T1 tile) counts; T1 of the completed ones)
                    int my_t = -1, my_rt = 0;
                    const int x = x0 + lane;
                    if (x < nn * nrt) {
                        const int i = x / nrt, rt = rtA + (x - i * nrt);
                        int nrow = 0;   // this unit's row tiles inside T1 tile rt
                        for (int j = 0; j < cn; ++j) nrow += (trow(r0 + j) >> 1) == rt ? 1 : 0;
                        int t;
                        unsigned add;
                        if (sp >= 0) { t = NMAX * sp + i; add = static_cast<unsigned>(nrow); }
                        else { const unsigned ev = T.ent[ebb + i]; t = static_cast<int>(ev & 511u); add = static_cast<unsigned>(__popc(ev >> 9) * nrow); }
                        unsigned nv = 0u;
                        for (int s2 = 0; s2 < pTK(P); ++s2)
                            nv += static_cast<unsigned>(small ? static_cast<int>(T.sids[t * pTK(P) + s2]) : __ldcg(P.ids + t * pTK(P) + s2)) <
                                  static_cast<unsigned>(NEXP) ? 1u : 0u;
                        const unsigned need = 2u * (nv + (pSHW(P) ? 1u : 0u));
                        if (atom_add_acq_rel(tc + t * NRT + rt, add) + add == need) { my_t = t; my_rt = rt; }
                    }
                    if (lane == 0 && x0 == 0) xt.rec(XT_END(59));
                    for (unsigned m = __ballot_sync(0xffffffffu, my_t >= 0); m; m &= m - 1u) {
                        const int src = __ffs(m) - 1;
                        const int tt = __shfl_sync(0xffffffffu, my_t, src), rr2 = __shfl_sync(0xffffffffu, my_rt, src);
                        t1_pair(P, T, small, rr2, tt, epoch, stg7, lane, &xt);
                    }
                    }
                    if (lane == 0) xt.rec(XT_BEG(59), 0u);
                }
                if (lane == 0) { q[T1Q + 2] = c; xt.rec(XT_END(59)); }
                __syncwarp();
            }
        }
        if (!t1l && !t1ll && TPB2_RTPUB && !TPB2_T1W && lane == 0) {   // ---- row-tile count publisher: FC2 units queued by the epilogue (rg = d.y % nrg)
            volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
            int c = 0;
            for (;;) {
                const int n = q[T1Q];
                if (c == n) {
                    if (q[T1Q + 1] && c == q[T1Q]) break;   // (epilogue done, everything published)
                    __nanosleep(64);   // (warp 7 shares an SM sub-partition with epilogue warps 3 / 11)
                    continue;
                }
                constexpr bool T1CNT = TPB2_T1CNT && !PR && NRT == 16 && NRT2U == 32 && T1Q <= 127;   // (<= 2 per tile per unit: bytes hold T1Q units)
                unsigned rn[NRT];
                uint64_t clo = 0ull, chi = 0ull;
#pragma unroll
                for (int i = 0; i < NRT; ++i) rn[i] = 0u;
                int hp[8], nhp = 0;   // (HPUB: FC1 h publishes of this batch: global panel | shared << 30)
                for (; c < n; ++c) {
                    const int qv = q[c % T1Q];
                    if (TPB2_HPUB && qv < 0) {
                        if (nhp == 8) break;   // (the rest in the next batch)
                        hp[nhp++] = qv & 0x7FFFFFFF;
                        continue;
                    }
                    const int r0 = TPB2_FC2VAR ? (qv & 0xFFFFFF) % NRT2U : (qv % nrg) * fc2g, cn = TPB2_FC2VAR ? (qv >> 24) : fc2g;
                    if (T1CNT) {   // row tiles r0 .. r0 + cn - 1 -> 2-bit counts per 256-row tile -> byte fields (tiles 0-7 | 8-15)
                        const uint32_t m = ((1u << cn) - 1u) << r0, c2 = (m & 0x55555555u) + ((m >> 1) & 0x55555555u);
                        auto spread = [](uint32_t x16) {
                            uint64_t x = x16;
                            x = (x | (x << 24)) & 0x000000FF000000FFull;
                            x = (x | (x << 12)) & 0x000F000F000F000Full;
                            return (x | (x << 6)) & 0x0303030303030303ull;
                        };
                        clo += spread(c2 & 0xFFFFu); chi += spread(c2 >> 16);
                        continue;
                    }
                    int ij[8];   // T1 tile (256 rows) of this CTA's row tiles of the unit
#pragma unroll
                    for (int j = 0; j < 8; ++j) ij[j] = j < cn ? (trow(r0 + j) >> 1) : -1;
#pragma unroll
                    for (int i = 0; i < NRT; ++i) {
                        unsigned a = 0u;
#pragma unroll
                        for (int j = 0; j < 8; ++j) a += ij[j] == i ? 1u : 0u;
                        rn[i] += a;
                    }
                }
                q[T1Q + 2] = c;
                if (!(TPB2_PROBE & 512)) fence_acq_rel();   // (the queued units' outp / shout rows: observed through the epilogue's barrier + queue entry)
#pragma unroll
                for (int i = 0; i < NRT; ++i) {
                    const unsigned r = T1CNT ? static_cast<unsigned>(((i < 8 ? clo : chi) >> (8 * (i & 7))) & 0xFFull) : rn[i];
                    if (r) red_relaxed_add(P.rt_cnt + par * NRT + i, r);
                }
                for (int i = 0; i < nhp; ++i) {   // (HPUB: the h images of these items, released by the fence above)
                    const int gp = hp[i] & 0x3FFFFFFF;
                    const unsigned need = (hp[i] & 0x40000000) ? static_cast<unsigned>(NT1S) : static_cast<unsigned>(NT1);
                    if (atom_add_acq_rel(P.pan_cnt + par * MAXP + gp, 1u) == need - 1u) {
                        unsigned* rl = P.rdy + par * (MAXP + 32);
                        st_release(rl + atomicAdd(rl + MAXP, 1u), static_cast<unsigned>(gp + 1));
                    }
                }
                xt.rec(XT_PT(58), static_cast<uint32_t>(n));
            }
        }
        __syncwarp();
    } else if (warp == 7) {
        // ======================================================== scale copier: SFA / SFB images -> TMEM
        if (lane == 0) {
            int fills = 0, w = 0;
            for (;;) {
                const int slot = w % WR;
                mbar_wait(bar(B_WF + slot), (w / WR) & 1);
                const int4 d = ring[slot];
                mbar_arrive(bar(B_WE + slot));
                ++w;
                if (d.x < 0) break;
                const int nst = TPB2_SFCP_MMA ? 0 : ((d.x == 1 || d.x == 3) ? d.w - d.z : f2n(d) * NSTG2);
                for (int j = 0; j < nst; ++j) {
                    const int st = fills % NS;
                    mbar_wait(bar(B_FULL + st), (fills / NS) & 1);
                    tc_fence_after();
                    const uint32_t tsf = tmem + TSF0 + st * TSF_STRIDE;
                    if (!(TPB2_PROBE & 1)) {
#pragma unroll
                        for (int c = 0; c < KC; ++c) tc_cp_sf(tsf + 4 * c, desc_sf(sb + OFF_SFA + (st * KC + c) * 512));
                        tc_cp_sf(tsf + TSF_B, desc_sf(sb + OFF_SFB + st * 512));
                    }
                    tc_commit(bar(B_SFR + st));
                    ++fills;
                }
            }
        }
        __syncwarp();
    } else if (warp == 6) {
        // ======================================================== B operand by bulk copies: FC1 x panel images / FC2 h panel images
        int fills = 0, w = 0, hub = 0;   // (hub: H1U FC2 units loaded)
        int rfi = 0;   // (TPB2_SFBI) routed FC1 items seen: item buffer rfi & 1
        bool x_ready = false;
        // (PAIR: this CTA's token rows rb .. rb + nb - 1 of a panel padded to 16; the SFB images keep every token)
        constexpr int NPR = PR ? 16 : 8;
        for (;;) {
            const int slot = w % WR;
            if (PR && !lead) mbar_wait_cl(bar(B_WF + slot), (w / WR) & 1); else mbar_wait(bar(B_WF + slot), (w / WR) & 1);
            const int4 d = ring[slot];
            use4(d);
            __syncwarp();
            if (lane == 0) we_arrive(slot);
            ++w;
            if (d.x < 0) break;
            if (lane == 0) xt.rec(XT_BEG(30), xd0(d), xd1(d));
            if (d.x == 1 || d.x == 3) {
                if (!x_ready && TPB2_GATHER) {   // the flags loaded before the tables (acquire): usually every token is in already
                    bool all = true;
#pragma unroll
                    for (int i = 0; i < 16; ++i) all &= xv[i] >= 8u;
                    if (__all_sync(0xffffffffu, all)) {
                        if (TPB2_XRLX) fence_acq_rel();   // (relaxed flag loads: acquire by fence)
                        if (lane == 0) fence_proxy_async_global();   // generic writes -> TMA gathers (async proxy)
                        if (DBGP && lane == 0) DBGP[b * 32 + 8] = gtime();
                        x_ready = true;
                    }
                }
                if (lane == 0) xt.rec(XT_PT(31));   // (x ready)
                if (!x_ready) {   // every token quantized and scattered (8 units each): all of a lane's polls in flight together
                    const unsigned* xc = P.xflag + par * MAXM;
                    for (;;) {
                        bool all = true;
                        for (int t0 = lane; t0 < M; t0 += 32 * 16) {
                            unsigned v[16];
#pragma unroll
                            for (int i = 0; i < 16; ++i) v[i] = t0 + 32 * i < M ? ld_relaxed(xc + t0 + 32 * i) : 8u;
#pragma unroll
                            for (int i = 0; i < 16; ++i) all &= v[i] >= 8u;
                        }
                        if (__all_sync(0xffffffffu, all)) break;
                    }
                    fence_acq_rel();
                    if (lane == 0) fence_proxy_async_global();   // generic writes -> bulk copies (async proxy)
                    if (DBGP && lane == 0) DBGP[b * 32 + 8] = gtime();
                    x_ready = true;
                }
                if (TPB2_GATHER && d.x == 1) {   // routed: the panel's token rows gathered from xq / xsf2 (lane g < npad / 4: rows 4 g .. 4 g + 3)
                    const int p = d.y / NT1U;
                    int e, n, eb;
                    panel_of(T, p, e, n, eb);
                    const int npad = (n + NPR - 1) & ~(NPR - 1), nb = PR ? npad / 2 : npad, rb = PR ? static_cast<int>(crank) * nb : 0, ng = nb >> 2;
                    // lane r < npad: token of panel row r (padding rows repeat the last); lane g < npad / 4 gathers rows 4 g .. 4 g + 3.  The SFB rows
                    // go by one 16-B bulk copy per lane (a gather4 of 16-B rows is 64 B: TMA destinations need 128-B alignment)
                    const int tok = lane < npad ? static_cast<int>(T.ent[eb + min(lane, n - 1)] & 511u) : 0;
                    int tr[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) tr[i] = __shfl_sync(0xffffffffu, tok, (rb + 4 * lane + i) & 31);
                    // SFB rows: generic 16-B loads two stages ahead, stored into the stage's SFB image before the stage's arrive (+ proxy fence)
                    const uint4* srow = reinterpret_cast<const uint4*>(P.xsf2 + static_cast<size_t>(tok) * NSTG1 * 16);
                    uint4 q0 = make_uint4(0u, 0u, 0u, 0u), q1 = q0;
                    constexpr int SB = TPB2_SFBI_SB;   // (SFBI) stages per SFB batch: one proxy fence per batch, the next batch's loads in flight
                    uint4 qb[SB];
                    const uint32_t ib = static_cast<uint32_t>(OFF_SFBI + (rfi & 1) * (NSTG1 * 512));
                    if (TPB2_SFBI) {
                        ++rfi;
#pragma unroll
                        for (int i = 0; i < SB; ++i) qb[i] = (lane < npad && d.z + i < d.w) ? __ldcg(srow + d.z + i) : make_uint4(0u, 0u, 0u, 0u);
                    } else if (lane < npad) { q0 = __ldcg(srow + d.z); if (d.z + 1 < d.w) q1 = __ldcg(srow + d.z + 1); }
                    for (int s = d.z; s < d.w; ++s) {
                        if (TPB2_SFBI && ((s - d.z) % SB) == 0) {   // batch s .. s + SB - 1 -> the item buffer, one fence; next batch's loads
                            if (lane < npad) {
#pragma unroll
                                for (int i = 0; i < SB; ++i)
                                    if (s + i < d.w) *reinterpret_cast<uint4*>(smem + ib + (s + i) * 512 + lane * 16) = qb[i];
                                fence_proxy_async_smem();
                            }
#pragma unroll
                            for (int i = 0; i < SB; ++i) qb[i] = (lane < npad && s + SB + i < d.w) ? __ldcg(srow + s + SB + i) : make_uint4(0u, 0u, 0u, 0u);
                            __syncwarp();
                        }
                        const int st = fills % NS;
                        if (fills >= NS) mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
                        if (!TPB2_SFBI) {
                        const uint4 cur = q0;
                        q0 = q1;
                        if (lane < npad && s + 2 < d.w) q1 = __ldcg(srow + s + 2);
                        if (lane < npad && !(TPB2_PROBE & 4) && !(TPB2_PROBE & 128)) {   // (PROBE 128: no SFB store + proxy fence, gathers kept)
                            *reinterpret_cast<uint4*>(smem + OFF_SFB + st * 512 + lane * 16) = cur;
                            fence_proxy_async_smem();
                        }
                        }
                        __syncwarp();
                        if (lane == 0) mbar_expect_tx(bar(B_FULL + st), (TPB2_PROBE & 4) ? 0 : KC * nb * 128);
                        if (DBGP && lane == 0 && fills == 0) DBGP[b * 32 + 29] = gtime();
                        if (lane < ng && !(TPB2_PROBE & 4)) {
#pragma unroll
                            for (int c = 0; c < KC; ++c)
                                tma_gather4(sb + OFF_B + st * B_BYTES + c * nb * 128 + lane * 512, &P.mapxq, (KC * s + c) * 128, tr[0], tr[1], tr[2], tr[3],
                                            bar(B_FULL + st));
                        }
                        ++fills;
                    }
                } else if (lane == 0) {
                    if (d.x == 1) {   // routed: KC chunks of the panel's x image per stage (!GATHER: 1-CTA only)
                        const int p = d.y / NT1U;
                        int e, n, eb;
                        panel_of(T, p, e, n, eb);
                        const int npad = (n + 7) & ~7;
                        for (int s = d.z; s < d.w; ++s) {
                            const int st = fills % NS;
                            if (fills >= NS) mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
                            mbar_expect_tx(bar(B_FULL + st), KC * npad * 128 + 512);
#pragma unroll
                            for (int c = 0; c < KC; ++c)
                                bulk_g2s(sb + OFF_B + st * B_BYTES + c * npad * 128, P.xg + ((static_cast<size_t>(p) * NCH1 + KC * s + c) * NMAX) * 128,
                                         npad * 128, bar(B_FULL + st));
                            bulk_g2s(sb + OFF_SFB + st * 512, P.xgsf + (static_cast<size_t>(p) * NSTG1 + s) * 512, 512, bar(B_FULL + st));
                            ++fills;
                        }
                    } else {   // shared: chunk s of the shared panel's per-128 x image per stage (SFB image of the chunk pair)
                        const int sp = d.y / NT1SU, n = min(NMAX, M - NMAX * sp), npad = (n + NPR - 1) & ~(NPR - 1);
                        const int nb = PR ? npad / 2 : npad, rb = PR ? static_cast<int>(crank) * nb : 0;
                        for (int s = d.z; s < d.w; ++s) {
                            const int st = fills % NS;
                            if (fills >= NS) mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
                            if (shh >= 0) {   // (SHGU: x chunks 2 s, 2 s + 1 and their SFB image s)
                                mbar_expect_tx(bar(B_FULL + st), 2 * nb * 128 + 512);
                                for (int h2 = 0; h2 < 2; ++h2)
                                    bulk_g2s(sb + OFF_B + st * B_BYTES + h2 * nb * 128, P.xgs + ((static_cast<size_t>(sp) * NCH1 + 2 * s + h2) * NMAX + rb) * 128, nb * 128, bar(B_FULL + st));
                                bulk_g2s(sb + OFF_SFB + st * 512, P.xgsfs + (static_cast<size_t>(sp) * NSTG1 + s) * 512, 512, bar(B_FULL + st));
                                ++fills;
                                continue;
                            }
                            mbar_expect_tx(bar(B_FULL + st), nb * 128 + 512);
                            bulk_g2s(sb + OFF_B + st * B_BYTES, P.xgs + ((static_cast<size_t>(sp) * NCH1 + s) * NMAX + rb) * 128, nb * 128, bar(B_FULL + st));
                            bulk_g2s(sb + OFF_SFB + st * 512, P.xgsfs + (static_cast<size_t>(sp) * NSTG1 + (s >> 1)) * 512, 512, bar(B_FULL + st));
                            ++fills;
                        }
                    }
                }
                fills = warp_uniform(static_cast<uint32_t>(fills));
                if (lane == 0) xt.rec(XT_END(30));
            } else {
                const int p = f2p(d), cn2 = f2n(d);
                int n;
                const uint8_t* himg;
                const uint8_t* hsf;
                if (d.x == 2) {
                    int e, eb;
                    panel_of(T, p, e, n, eb);
                    himg = P.himg + static_cast<size_t>(p) * NCH2 * NMAX * 128; hsf = P.hsf + static_cast<size_t>(p) * NSTG2 * 512;
                } else {
                    const int sp = p - Ptot;
                    n = min(NMAX, M - NMAX * sp);
                    himg = P.hsimg + static_cast<size_t>(sp) * NCH2 * NMAX * 128; hsf = P.hssf + static_cast<size_t>(sp) * NSTG2 * 512;
                }
                const int npad = (n + NPR - 1) & ~(NPR - 1), nb = PR ? npad / 2 : npad, rb = PR ? static_cast<int>(crank) * nb : 0;
                const bool h1u = TPB2_H1U && !PR;
                if (lane == 0) {
                    if (f2s || fc2l || TPB2_FC2RLX) {   // the panel's h: all its FC1 items in (acquire; FC2RLX: the producer read its slot relaxed)
                        const unsigned long long tw = DBGP ? ctime() : 0ull;
                        wait_ge(P.pan_cnt + par * MAXP + p, d.x == 2 ? static_cast<unsigned>(NT1) : static_cast<unsigned>(NT1S));
                        if (DBGP) DBGP[b * 32 + 21] += ctime() - tw;
                    }
                    fence_proxy_async_global();   // (generic h writes -> bulk copies; FC2STATIC=0: the producer acquired the panel's h flag)
                    if (h1u) {   // H1U: the unit's h into buffer u (the B / SFB ring regions; the CTA's FC1 stages all consumed first)
                        const int u = hub & 1;
                        if (hub == 0 && fills > 0) mbar_wait(bar(B_EMPTY + (fills - 1) % NS), static_cast<uint32_t>((fills - 1) / NS) & 1u);
                        if (hub >= 2) mbar_wait(bar(B_HE0 + u), static_cast<uint32_t>((hub - 2) >> 1) & 1u);
                        mbar_expect_tx(bar(B_HF0 + u), NSTG2 * (KC * nb * 128 + 512));
                        for (int s2 = 0; s2 < NSTG2; ++s2) {
#pragma unroll
                            for (int c = 0; c < KC; ++c)
                                bulk_g2s(sb + OFF_B + u * (NSTG2 * KC * NMAX * 128) + (KC * s2 + c) * nb * 128, himg + (static_cast<size_t>(KC * s2 + c) * NMAX + rb) * 128,
                                         nb * 128, bar(B_HF0 + u));
                            bulk_g2s(sb + OFF_SFB + (u * NSTG2 + s2) * 512, hsf + static_cast<size_t>(s2) * 512, 512, bar(B_HF0 + u));
                        }
                    }
                    for (int f = fills; f < fills + cn2 * NSTG2; ++f) {   // the panel's h once per row tile (L2)
                        const int st = f % NS, s = (f - fills) % NSTG2;
                        if (f >= NS) mbar_wait(bar(B_EMPTY + st), ((f / NS) - 1) & 1);
                        const bool nob = h1u || (TPB2_PROBE & 4) || ((TPB2_PROBE & 64) && f >= fills + NSTG2);   // (64: h of the unit's first row tile only)
                        mbar_expect_tx(bar(B_FULL + st), nob ? 0 : KC * nb * 128 + 512);
                        if (!nob) {
#pragma unroll
                            for (int c = 0; c < KC; ++c)
                                bulk_g2s(sb + OFF_B + st * B_BYTES + c * nb * 128, himg + (static_cast<size_t>(KC * s + c) * NMAX + rb) * 128, nb * 128, bar(B_FULL + st));
                            bulk_g2s(sb + OFF_SFB + st * 512, hsf + static_cast<size_t>(s) * 512, 512, bar(B_FULL + st));
                        }
                    }
                }
                __syncwarp();
                if (lane == 0) xt.rec(XT_END(30));
                fills += cn2 * NSTG2;
                if (h1u) ++hub;
            }
        }
        __syncwarp();
    } else if (warp == 1) {
        // ======================================================== MMA issuer (converged warp, one elected lane)
        // tcgen05.mma operands must be warp-uniform (uniform registers): TMEM base, descriptor fields via shuffles (else every MMA
        // issue gets an R2UR loop -- measured ~48 vs ~25 ns per MMA)
        const uint32_t tmem_u = warp_uniform(tmem);
        // (TPB2_LEANMMA 2) stage-0 operands, computed once: stage st adds st * stride
        const uint64_t adesc0 = desc_sw128(sb + OFF_A), bdesc0 = desc_sw128(sb + OFF_B), sfadesc0 = desc_sf(sb + OFF_SFA), sfbdesc0 = desc_sf(sb + OFF_SFB);
        const uint32_t bfull0 = bar(B_FULL), bempty0 = bar(B_EMPTY), tsf0 = tmem_u + TSF0, bfullp0 = bar(B_FULLP);
        const uint32_t fullp_lead = PR ? peer_addr(bar(B_FULLP), 0u) : 0u;   // (PAIR peer: the leader's FULLP barriers)
        int fills = 0, w = 0, kacc = 0;   // accumulators issued so far: buffer kacc % ACC, use kacc / ACC
        int rfi = 0;   // (TPB2_SFBI) routed FC1 items seen: SFB item buffer rfi & 1
        const uint64_t sfbidesc0 = desc_sf(sb + OFF_SFBI);
        int hum = 0;   // (H1U: FC2 units issued -- h buffer hum & 1)
        int ncp = 0;   // (PROBE 16: SF copies only while the ring's TMEM SF slots are first filled -- later MMAs reuse stale, finite scales)
        auto cpsf = [&](uint32_t taddr, uint64_t desc) { if (!(TPB2_PROBE & 16) || ncp < 6 * NS) tc_cp_sf(taddr, desc); ++ncp; };
        unsigned long long acc_w1 = 0, acc_w2 = 0, acc_i1 = 0;   // (dbg) FC1 / FC2 data waits, FC1 issue time
        for (;;) {
            const int slot = w % WR;
            if (lane == 0) xt.rec(XT_PT(72));
            if (PR && !lead) mbar_wait_cl(bar(B_WF + slot), (w / WR) & 1); else mbar_wait(bar(B_WF + slot), (w / WR) & 1);
            int4 d = ring[slot];
            d.x = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.x))); d.y = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.y)));
            d.z = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.z))); d.w = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.w)));
            use4(d);
            __syncwarp();
            if (lane == 0) we_arrive(slot);
            ++w;
            if (d.x < 0) {
                if (DBGP && lane == 0) { DBGP[b * 32 + 22] = acc_w1; DBGP[b * 32 + 16] = acc_w2; DBGP[b * 32 + 23] = acc_i1; }
                break;
            }
            if (PR && !lead) {   // PAIR peer: forward each of the descriptor's stages (this CTA's operands landed) to the leader's FULLP
                const int ntot = (d.x & 1) ? d.w - d.z : f2n(d) * NSTG2;
                for (int j = 0; j < ntot; ++j, ++fills) {
                    const int st = fills % NS;
                    if (lane == 0) xt.rec(XT_BEG(24));
                    mbar_wait(bfull0 + 8 * st, static_cast<uint32_t>(fills / NS) & 1u);
                    if (lane == 0) { xt.rec(XT_END(24)); mbar_arrive_remote_relaxed(fullp_lead + 8 * st); }
                }
                __syncwarp();
                continue;
            }
            if (lane == 0) xt.rec(XT_PT(71));
            const bool sfbi = TPB2_SFBI && d.x == 1;   // (this routed FC1 item's SFB rows: item buffer rfi & 1, stage d.z + j)
            const uint64_t sfbi0 = sfbidesc0 + static_cast<uint64_t>((((rfi & 1) * NSTG1) + d.z) * (512 >> 4));
            if (sfbi) ++rfi;
            const int n = panel_n(d);
            const uint32_t npad = warp_uniform(static_cast<uint32_t>(PR ? (n + 15) & ~15 : (n + 7) & ~7));
            const uint32_t nbr = PR ? npad / 2 : npad;   // (B rows in this CTA's smem)
            const bool shk = d.x >= 3;   // shared expert: e4m3 A
            const uint32_t id0 = idesc_f4f8(static_cast<int>(npad), 0, shk ? 0u : 5u);
            const bool fc1 = (d.x & 1) != 0;
            const int nacc = fc1 ? 1 : f2n(d), nst = fc1 ? d.w - d.z : NSTG2;
            for (int a = 0; a < nacc; ++a) {   // FC1: one accumulator; FC2: one per row tile of the group
                const int ab = kacc % ACC;
                if (kacc >= ACC) { if (lane == 0) xt.rec(XT_BEG(20)); mbar_wait(bar(B_ACCE + ab), ((kacc / ACC) - 1) & 1); if (lane == 0) xt.rec(XT_END(20)); }
                if (lane == 0) xt.rec(XT_BEG(21), xd0(d), xd1(d) | static_cast<uint32_t>(a));
                tc_fence_after();
                const uint32_t dacc = tmem_u + ab * ACC_COLS;
#if !TPB2_MPAIR
                if (TPB2_LEANMMA == 3) {
                    const int s0 = fills & (NS - 1);
                    const uint32_t ph0 = static_cast<uint32_t>(fills / NS) & 1u, bstep = (npad * 128) >> 4;
                    const int sel = s0 + (d.x == 3 ? NS : 0);
                    static_assert(NS == 4 || TPB2_LEANMMA != 3, "LEANMMA 3 dispatch");
                    switch (sel) {
                        case 0: mma_stages<0, false>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                        case 1: mma_stages<1, false>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                        case 2: mma_stages<2, false>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                        case 3: mma_stages<3, false>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                        case 4: mma_stages<0, true>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                        case 5: mma_stages<1, true>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                        case 6: mma_stages<2, true>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                        default: mma_stages<3, true>(nst, d.z, ph0, dacc, id0, bstep, sb, tsf0, bfull0, bempty0); break;
                    }
                    fills += nst;
                } else if (TPB2_LEANMMA == 2) {   // the per-stage loop as lean as the clean bench: per-item operands hoisted, stage / parity incremental
                    int st = fills % NS;
                    uint32_t ph = static_cast<uint32_t>(fills / NS) & 1u;
                    const uint64_t bstep = static_cast<uint64_t>((nbr * 128) >> 4);
                    if (d.x == 3) {
                        for (int j = 0; j < nst; ++j) {
                            if (lane == 0) xt.rec(XT_BEG(22));
                            mbar_wait(bfull0 + 8 * st, ph);
                            if (lane == 0) xt.rec(XT_END(22));
                            if (PR) {   // (the peer's B / scales landed, forwarded)
                                if (lane == 0) xt.rec(XT_BEG(23));
                                mbar_wait(bfullp0 + 8 * st, ph);   // (relaxed forwarded arrivals: phase only)
                                if (lane == 0) xt.rec(XT_END(23));
                            }
                            tc_fence_after();
                            const uint32_t tsf = tsf0 + st * TSF_STRIDE;
                            const uint64_t ad0 = adesc0 + static_cast<uint64_t>(st * (A_BYTES >> 4)), bd0 = bdesc0 + static_cast<uint64_t>(st * (B_BYTES >> 4));
                            if (elect_one()) {
#pragma unroll
                                for (int c = 0; c < KC; ++c) cpsf(tsf + 4 * c, sfadesc0 + static_cast<uint64_t>((st * KC + c) * (512 >> 4)));
                                cpsf(tsf + TSF_B, sfbdesc0 + static_cast<uint64_t>(st * (512 >> 4)));
                                const uint32_t sfb = tsf + TSF_B + 2 * ((d.z + j) & 1);
                                if (shh >= 0) {   // SHGU: chunks 2 j, 2 j + 1 of this CTA's chain (A halves, B chunks, SFA / SFB entries c), K order
#pragma unroll
                                    for (int c = 0; c < 2; ++c)
#pragma unroll
                                        for (int kb = 0; kb < 4; ++kb)
                                            tc_mma(dacc + 32 * shh, ad0 + c * (16384 >> 4) + 2 * kb, bd0 + c * bstep + 2 * kb, id0 | (kb << 4) | (kb << 29),
                                                   tsf + 4 * c, tsf + TSF_B + 2 * c, (j | c | kb) != 0);
                                } else
#pragma unroll
                                for (int kb = 0; kb < 4; ++kb) {
                                    const uint32_t idk = id0 | (kb << 4) | (kb << 29), acc = (j | kb) != 0;
                                    tc_mma(dacc, ad0 + 2 * kb, bd0 + 2 * kb, idk, tsf, sfb, acc);
                                    tc_mma(dacc + 32, ad0 + (16384 >> 4) + 2 * kb, bd0 + 2 * kb, idk, tsf + 4, sfb, acc);
                                }
                                tc_commit(bempty0 + 8 * st);
                            }
                            __syncwarp();
                            if (++st == NS) { st = 0; ph ^= 1u; }
                        }
                    } else if ((TPB2_FC2PAIR && M > TPB2_FC2PAIR_OFFM && !fc1 && nst == 2) || (TPB2_FC1PAIR && fc1 && (nst & 1) == 0)) {
                      const bool h1u = TPB2_H1U && !PR && !fc1;
                      const int hu = hum & 1;
                      if (h1u && a == 0) mbar_wait(bar(B_HF0 + hu), static_cast<uint32_t>(hum >> 1) & 1u);   // (the unit's h buffer loaded)
                      const uint64_t hbd = desc_sw128(sb + OFF_B + hu * (NSTG2 * KC * NMAX * 128));
                      for (int jp = 0; jp < nst; jp += 2) {
                        int st1 = st + 1;
                        uint32_t ph1 = ph;
                        if (st1 == NS) { st1 = 0; ph1 ^= 1u; }
                        if (lane == 0) xt.rec(XT_BEG(22));
                        mbar_wait(bfull0 + 8 * st, ph);
                        mbar_wait(bfull0 + 8 * st1, ph1);
                        if (PR) { mbar_wait(bfullp0 + 8 * st, ph); mbar_wait(bfullp0 + 8 * st1, ph1); }
                        if (lane == 0) xt.rec(XT_END(22));
                        tc_fence_after();
                        if (elect_one()) {
#pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                const int sx = h ? st1 : st;
                                const uint32_t tsf = tsf0 + sx * TSF_STRIDE;
                                const uint64_t ad0 = adesc0 + static_cast<uint64_t>(sx * (A_BYTES >> 4)),
                                               bd0 = h1u ? hbd + static_cast<uint64_t>((KC * (jp + h)) * bstep) : bdesc0 + static_cast<uint64_t>(sx * (B_BYTES >> 4));
#pragma unroll
                                for (int c = 0; c < KC; ++c) cpsf(tsf + 4 * c, sfadesc0 + static_cast<uint64_t>((sx * KC + c) * (512 >> 4)));
                                cpsf(tsf + TSF_B, sfbdesc0 + static_cast<uint64_t>((h1u ? hu * NSTG2 + jp + h : sx) * (512 >> 4)));
#pragma unroll
                                for (int c = 0; c < KC; ++c) {
                                    const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * bstep;
#pragma unroll
                                    for (int kb = 0; kb < 4; ++kb)
                                        tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + TSF_B + 2 * c, (jp | h | c | kb) != 0);
                                }
                            }
                            tc_commit(bempty0 + 8 * st);
                            tc_commit(bempty0 + 8 * st1);
                            if (h1u && a == nacc - 1 && jp + 2 >= nst) tc_commit(bar(B_HE0 + hu));   // (the unit's last MMAs: h buffer free)
                        }
                        __syncwarp();
                        st = st1 + 1; ph = ph1;
                        if (st == NS) { st = 0; ph ^= 1u; }
                      }
                    } else {
                        for (int j = 0; j < nst; ++j) {
                            if (lane == 0) xt.rec(XT_BEG(22));
                            mbar_wait(bfull0 + 8 * st, ph);
                            if (lane == 0) xt.rec(XT_END(22));
                            if (PR) {   // (the peer's B / scales landed, forwarded)
                                if (lane == 0) xt.rec(XT_BEG(23));
                                mbar_wait(bfullp0 + 8 * st, ph);   // (relaxed forwarded arrivals: phase only)
                                if (lane == 0) xt.rec(XT_END(23));
                            }
                            tc_fence_after();
                            const uint32_t tsf = tsf0 + st * TSF_STRIDE;
                            const uint64_t ad0 = adesc0 + static_cast<uint64_t>(st * (A_BYTES >> 4)), bd0 = bdesc0 + static_cast<uint64_t>(st * (B_BYTES >> 4));
                            if (elect_one()) {
#pragma unroll
                                for (int c = 0; c < KC; ++c) cpsf(tsf + 4 * c, sfadesc0 + static_cast<uint64_t>((st * KC + c) * (512 >> 4)));
                                cpsf(tsf + TSF_B, sfbi ? sfbi0 + static_cast<uint64_t>(j * (512 >> 4)) : sfbdesc0 + static_cast<uint64_t>(st * (512 >> 4)));
#pragma unroll
                                for (int c = 0; c < KC; ++c) {
                                    const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * bstep;
#pragma unroll
                                    for (int kb = 0; kb < 4; ++kb)
                                        tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + TSF_B + 2 * c, (j | c | kb) != 0);
                                }
                                tc_commit(bempty0 + 8 * st);
                            }
                            __syncwarp();
                            if (++st == NS) { st = 0; ph ^= 1u; }
                        }
                    }
                    fills += nst;
                } else if (TPB2_LEANMMA) {   // one stage per round, minimal per-stage work on the issuing warp (it paces the FC1 stream)
                    for (int j = 0; j < nst; ++j) {
                        const int st = fills % NS;
                        const unsigned long long tw0 = DBGP ? ctime() : 0ull;
                        mbar_wait(bar(B_FULL + st), (fills / NS) & 1);
                        if (!TPB2_SFCP_MMA) mbar_wait(bar(B_SFR + st), (fills / NS) & 1);
                        const unsigned long long tw1 = DBGP ? ctime() : 0ull;
                        if (DBGP) {
                            if (fc1) acc_w1 += tw1 - tw0; else acc_w2 += tw1 - tw0;
                            if (lane == 0) { if (fills == 0) DBGP[b * 32 + 9] = gtime(); if (d.x == 1) DBGP[b * 32 + 10] = gtime(); }
                        }
                        tc_fence_after();
                        const uint32_t tsf = tmem_u + TSF0 + st * TSF_STRIDE;
                        const uint64_t ad0 = desc_sw128(sb + OFF_A + st * A_BYTES), bd0 = desc_sw128(sb + OFF_B + st * B_BYTES);
                        if (elect_one()) {
                            if (TPB2_SFCP_MMA && !(TPB2_PROBE & 1)) {
#pragma unroll
                                for (int c = 0; c < KC; ++c) cpsf(tsf + 4 * c, desc_sf(sb + OFF_SFA + (st * KC + c) * 512));
                                cpsf(tsf + TSF_B, desc_sf(sb + OFF_SFB + st * 512));
                            }
                            if (!(TPB2_PROBE & 2)) {
                                if (d.x == 3) {
                                    const uint32_t sfb = tsf + TSF_B + 2 * ((d.z + j) & 1);
#pragma unroll
                                    for (int kb = 0; kb < 4; ++kb) {
                                        const uint32_t idk = id0 | (kb << 4) | (kb << 29), acc = (j | kb) != 0;
                                        tc_mma(dacc, ad0 + 2 * kb, bd0 + 2 * kb, idk, tsf, sfb, acc);
                                        tc_mma(dacc + 32, ad0 + (16384 >> 4) + 2 * kb, bd0 + 2 * kb, idk, tsf + 4, sfb, acc);
                                    }
                                } else {
#pragma unroll
                                    for (int c = 0; c < KC; ++c) {
                                        const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * ((npad * 128) >> 4);
#pragma unroll
                                        for (int kb = 0; kb < 4; ++kb)
                                            tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + TSF_B + 2 * c, (j | c | kb) != 0);
                                    }
                                    if (TPB2_PROBE & 16) {   // (timing probe: the stage's MMAs again, into the scale region's spare columns -- WRONG results)
#pragma unroll
                                        for (int c = 0; c < KC; ++c) {
                                            const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * ((npad * 128) >> 4);
#pragma unroll
                                            for (int kb = 0; kb < 4; ++kb)
                                                tc_mma(tmem_u + 3 * ACC_COLS + 32, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + TSF_B + 2 * c, 1u);
                                        }
                                    }
                                }
                            }
                            tc_commit(bar(B_EMPTY + st));
                        }
                        __syncwarp();
                        if (DBGP && fc1) acc_i1 += ctime() - tw1;
                        ++fills;
                    }
                } else
#endif
                for (int j = 0; j < nst;) {
                    // TPB2_MPAIR: two ring stages per issue round (16 MMAs between the waits / fence / commit)
                    const int two = (TPB2_MPAIR && j + 1 < nst) ? 2 : 1;
                    const int st0 = fills % NS, st1 = (fills + 1) % NS;
                    const unsigned long long tw0 = DBGP ? ctime() : 0ull;
                    mbar_wait(bar(B_FULL + st0), (fills / NS) & 1);
                    if (!TPB2_SFCP_MMA) mbar_wait(bar(B_SFR + st0), (fills / NS) & 1);
                    if (two == 2) {
                        mbar_wait(bar(B_FULL + st1), ((fills + 1) / NS) & 1);
                        if (!TPB2_SFCP_MMA) mbar_wait(bar(B_SFR + st1), ((fills + 1) / NS) & 1);
                    }
                    const unsigned long long tw1 = DBGP ? ctime() : 0ull;
                    if (fc1) acc_w1 += tw1 - tw0; else acc_w2 += tw1 - tw0;
                    if (DBGP && lane == 0) { if (fills == 0) DBGP[b * 32 + 9] = gtime(); if (d.x == 1) DBGP[b * 32 + 10] = gtime(); }
                    tc_fence_after();
                    const uint32_t tsfa[2] = {tmem_u + TSF0 + st0 * TSF_STRIDE, tmem_u + TSF0 + st1 * TSF_STRIDE};
                    const uint64_t ada[2] = {desc_sw128(sb + OFF_A + st0 * A_BYTES), desc_sw128(sb + OFF_A + st1 * A_BYTES)};
                    const uint64_t bda[2] = {desc_sw128(sb + OFF_B + st0 * B_BYTES), desc_sw128(sb + OFF_B + st1 * B_BYTES)};
                    if (elect_one()) {
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            if (h >= two || (TPB2_PROBE & 2)) break;
                            const int jj = j + h;
                            const uint32_t tsf = tsfa[h];
                            const uint64_t ad0 = ada[h], bd0 = bda[h];
                            if (TPB2_SFCP_MMA && !(TPB2_PROBE & 1)) {   // this stage's SFA / SFB images -> TMEM, ahead of its MMAs
                                const int st = h ? st1 : st0;
#pragma unroll
                                for (int c = 0; c < KC; ++c) cpsf(tsf + 4 * c, desc_sf(sb + OFF_SFA + (st * KC + c) * 512));
                                cpsf(tsf + TSF_B, desc_sf(sb + OFF_SFB + st * 512));
                            }
                            if (d.x == 3) {   // shared FC1: chunk s of the gate tile (-> columns 0..) and of the up tile (-> columns 32..), one B chunk
                                const uint32_t sfb = tsf + TSF_B + 2 * ((d.z + jj) & 1);
#pragma unroll
                                for (int kb = 0; kb < 4; ++kb) {
                                    const uint32_t idk = id0 | (kb << 4) | (kb << 29), acc = (jj | kb) != 0;
                                    if (shh != 1) tc_mma(dacc, ad0 + 2 * kb, bd0 + 2 * kb, idk, tsf, sfb, acc);
                                    if (shh != 0) tc_mma(dacc + 32, ad0 + (16384 >> 4) + 2 * kb, bd0 + 2 * kb, idk, tsf + 4, sfb, acc);
                                }
                            } else {
#pragma unroll
                                for (int c = 0; c < KC; ++c) {
                                    const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * ((npad * 128) >> 4);
#pragma unroll
                                    for (int kb = 0; kb < 4; ++kb)
                                        tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + TSF_B + 2 * c, (jj | c | kb) != 0);
                                }
                            }
                        }
                        tc_commit(bar(B_EMPTY + st0));
                        if (two == 2) tc_commit(bar(B_EMPTY + st1));
                    }
                    __syncwarp();
                    if (DBGP && fc1) acc_i1 += ctime() - tw1;
                    fills += two;
                    j += two;
                }
                if (elect_one()) tc_commit(bar(B_ACCF + ab));
                __syncwarp();
                if (lane == 0) xt.rec(XT_END(21));
                ++kacc;
            }
            if (TPB2_H1U && !PR && !fc1) ++hum;
        }
    } else {
        // ======================================================== epilogue: warps 2..5 (group 0: token columns 0..15) and 8..11 (group 1:
        // 16..31); TMEM lanes 32 q .. 32 q + 31, q = warp & 3.  Named barriers: 1 + g (the group's 128 threads), 3 (both groups)
        const int q = warp & 3, L = 32 * q + lane, g = warp >= 8 ? 1 : 0, tb = 16 * g, gt = tid - (g ? 256 : 64);
        const bool up_lane = (lane & 8) == 0;
        const bool pub = warp == 2 && lane == 0;   // the publishing thread
        const int frow = T.lanes[L];

        __nv_bfloat16* stg = T.stg + g * 16 * 128;   // FC2 transpose staging [16 tokens][128 rows]
        int w = 0, kacc = 0, csk = 0;   // (csk: CSPLIT routed K-half pieces so far -- partial slot csk & 1)
        unsigned rt_n[NRT];   // (pub) FC2 row tiles stored per 256-row T1 tile, published once at the end
#pragma unroll
        for (int i = 0; i < NRT; ++i) rt_n[i] = 0u;
        bool first_fc1 = true;
        int nfl = 0;   // (FC2L: FC2 units processed)
        int dry = TPB2_WARM ? 2 : 0;   // dry passes left: 2 = routed FC1 (panel 0's tile 0, whole), 1 = FC2 (panel 0, row group 0)
        for (;;) {
            int4 d;
            if (TPB2_WARM_WAIT == 2 && dry == 2) {
                mbar_wait(bar(B_WF + 0), 0);
                const int x0 = ring[0].x;
                if (x0 == 1 || x0 == 3) mbar_wait(bar(B_FULL + 1), 0);   // (FC1 first: warm up while its stages stream)
                else dry = 0;                                            // (FC2 / nothing first: no dry pass on the critical path)
            }
            const bool isdry = dry != 0;
            if (isdry) {
                if (TPB2_WARM_WAIT == 1 && dry == 2) mbar_wait(bar(B_WF + 0), 0);   // (the producer has started: its code is in)
                d = dry == 2 ? make_int4(1, 0, 0, NSTG1) : make_int4(2, 0, 1, 0);   // (FC2VAR: d.z = 1 row tile)
                --dry;
            } else {
                if (TPB2_EPIFIRST && pub) *reinterpret_cast<volatile int*>(T.misc + 26) = w;   // (descriptors 0 .. w - 1 fully processed)
                const int slot = w % WR;
                if (PR && !lead) mbar_wait_cl(bar(B_WF + slot), (w / WR) & 1); else mbar_wait(bar(B_WF + slot), (w / WR) & 1);
                d = ring[slot];
                use4(d);
                __syncwarp();
                if (lane == 0) we_arrive(slot);
                ++w;
            }
            if (d.x < 0) {
                if (pub) xt.rec(XT_PT(49));
                if (pub) {
                    if (!TPB2_RTPUB || TPB2_T1W) {
                        fence_acq_rel();
#pragma unroll
                        for (int i = 0; i < NRT; ++i) if (rt_n[i]) red_relaxed_add(P.rt_cnt + par * NRT + i, rt_n[i]);
                    }
                    if (TPB2_T1W || TPB2_RTPUB) { __threadfence_block(); reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q)[T1Q + 1] = 1; }
                }
                break;
            }
            int p, e = 0, n, eb = 0;   // routed: panel; shared: shared panel sp (tokens 32 sp ..)
            if (d.x == 1) { p = d.y / NT1U; panel_of(T, p, e, n, eb); }
            else if (d.x == 2) { p = f2p(d); panel_of(T, p, e, n, eb); }
            else if (d.x == 3) { p = d.y / NT1SU; n = min(NMAX, M - NMAX * p); }
            else { p = f2p(d) - Ptot; n = min(NMAX, M - NMAX * p); }
            const int npad = PR ? (n + 15) & ~15 : (n + 7) & ~7;
            const bool fc1 = (d.x & 1) != 0;
            const int nacc = fc1 ? 1 : f2n(d);
            for (int a = 0; a < nacc; ++a) {
                const int ab = kacc % ACC;
                const unsigned long long te0 = DBGP ? ctime() : 0ull;
                if (pub) xt.rec(XT_BEG(40), xd0(d), xd1(d) | static_cast<uint32_t>(a));
                if (!isdry) mbar_wait(bar(B_ACCF + ab), (kacc / ACC) & 1);
                if (pub) xt.rec(XT_END(40));
                const unsigned long long te1 = DBGP ? ctime() : 0ull;
                if (DBGP && pub && !fc1 && !isdry) DBGP[b * 32 + 18] += te1 - te0;
                if (DBGP && pub && d.x == 1 && first_fc1 && !isdry) { DBGP[b * 32 + 11] = gtime(); first_fc1 = false; }
                tc_fence_after();
                float v[16], vu[16];
                const uint32_t ta = tmem + (static_cast<uint32_t>(32 * q) << 16) + ab * ACC_COLS + tb;
                tc_ld16_wait(ta, v);                  // (columns past npad: unused garbage)
                if (d.x == 3) tc_ld16_wait(ta + 32, vu);   // shared FC1: the up tile's accumulator
                tc_fence_before();
                __syncwarp();
                if (!isdry) {
                    if (lane == 0) { if (PR && !lead) mbar_arrive_remote_relaxed(peer_addr(bar(B_ACCE + ab), 0u)); else mbar_arrive(bar(B_ACCE + ab)); }
                    ++kacc;
                }
                if (!fc1) {
                    if (TPB2_DLY_SITE == 4) dly_spin(TPB2_DLY_NS);
                    if (pub) xt.rec(XT_BEG(47), xd0(d), static_cast<uint32_t>(a));
                    // ---- FC2 row tile r: transpose through smem, 16-B rows of bf16 per (token, slot) / shared: per token of shout
                    const int r = trow(f2r0(d) + a);
                    const int orow = d.x == 4 ? L : frow;   // (the shared down weights are used in production row order)
                    if (fc2l) {   // FC2L: token 0's row tile outputs per panel -> smem [panel][128 rows]; the last unit T1s the 128 rows
                        __nv_bfloat16* t1a = T.stg;   // [8][128] bf16 (+ [128] results at 8 x 128)
                        const int pi2 = d.x == 2 ? p : Ptot + p;
                        if (g == 0 && !isdry) t1a[pi2 * 128 + orow] = __float2bfloat16_rn(v[0]);
                        if (!isdry && ++nfl == Pall) {
                            named_bar(1 + g, 128);
                            if (g == 0) {   // row gt: slots in order (bf16 weights), bf16 rounding, factor * r + shared, bf16 (== t1_emit)
                                float acc = 0.f;
                                for (int k = 0; k < pTK(P); ++k) {
                                    const int id = static_cast<int>(T.sids[k]);
                                    if (static_cast<unsigned>(id) >= static_cast<unsigned>(NEXP)) continue;
                                    int pk = 0;   // (M = 1: one panel per present expert, ascending ids -> the rank of id among them)
                                    for (int j2 = 0; j2 < pTK(P); ++j2) {
                                        const int idj = static_cast<int>(T.sids[j2]);
                                        if (static_cast<unsigned>(idj) >= static_cast<unsigned>(NEXP) || idj >= id) continue;
                                        bool first = true;
                                        for (int j3 = 0; j3 < j2; ++j3) first &= static_cast<int>(T.sids[j3]) != idj;
                                        pk += first ? 1 : 0;
                                    }
                                    acc = __fmaf_rn(__bfloat162float(t1a[pk * 128 + gt]), bf16r(T.srw[k]), acc);
                                }
                                const float y = __fmaf_rn(P.factor, bf16r(acc), __bfloat162float(t1a[Ptot * 128 + gt]));
                                t1a[8 * 128 + gt] = __float2bfloat16_rn(y);
                                named_bar(1, 128);
                                if (warp == 2 && lane < 22) {   // the one-shot messages of this 128-row half
                                    const uint32_t* w = reinterpret_cast<const uint32_t*>(t1a + 8 * 128 + 6 * lane);
                                    const bool full = lane < 21;
                                    const uint4 msg = make_uint4(w[0], full ? w[1] : 0u, full ? w[2] : 0u, epoch);
#pragma unroll
                                    for (int dv = 0; dv < NDEV; ++dv)
                                        st_ll(P.rs[(pRK(P) + dv) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(pRK(P), r >> 1, 0, (r & 1) * 22 + lane), msg);
                                }
                            }
                        }
                        if (pub) xt.rec(XT_END(47));
                        continue;
                    }
                    if (TPB2_FC2DIRECT) {
#pragma unroll 1
                        for (int i = 0; i < 16; ++i) {
                            if (tb + i >= n) break;
                            const __nv_bfloat16 hv = __float2bfloat16_rn(v[i]);
                            if (isdry) continue;
                            if (d.x == 4) { P.shout[static_cast<size_t>(NMAX * p + tb + i) * DIM + r * 128 + orow] = hv; continue; }
                            const unsigned ev = T.ent[eb + tb + i];
                            const int t = static_cast<int>(ev & 511u);
                            for (unsigned km = ev >> 9; km; km &= km - 1)
                                if (!(TPB2_PROBE & 1024)) P.outp[(static_cast<size_t>(t) * TOPK_MAX + (__ffs(km) - 1)) * OUTP_LD + r * 128 + orow] = hv;
                        }
                        if (pub) xt.rec(XT_END(47));
                        if (DBGP && pub && !isdry) DBGP[b * 32 + 19] += ctime() - te1;
                        continue;
                    }
#pragma unroll
                    for (int i = 0; i < 16; ++i) if (tb + i < n) stg[i * 128 + orow] = __float2bfloat16_rn(v[i]);
                    named_bar(1 + g, 128);
                    if (t1ll_on(P)) {   // T1LL: LL messages (rows 6 c .. 6 c + 5 of the row tile + the epoch) per (token i, message c)
                        uint4* ll = reinterpret_cast<uint4*>(P.tcnt + LL_OFF);
                        for (int x = gt; x < 16 * LLM; x += 128) {
                            const int i = x / LLM, c = x - i * LLM;
                            if (tb + i >= n) break;
                            const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + i * 128 + 6 * c);
                            const bool full = c < LLM - 1;
                            const uint4 msg = make_uint4(w[0], full ? w[1] : 0u, full ? w[2] : 0u, epoch);
                            if (isdry) continue;
                            if (d.x == 4) { st_llg(ll + static_cast<size_t>(NMAX * p + tb + i) * LL_TOK + (TOPK_MAX * 32 + r) * LLM + c, msg); continue; }
                            const unsigned ev = T.ent[eb + tb + i];
                            const int t = static_cast<int>(ev & 511u);
                            for (unsigned km = ev >> 9; km; km &= km - 1)
                                st_llg(ll + static_cast<size_t>(t) * LL_TOK + ((__ffs(km) - 1) * 32 + r) * LLM + c, msg);
                        }
                        named_bar(1 + g, 128);   // (stg reused by the next row tile)
                        if (pub) xt.rec(XT_END(47));
                        continue;
                    }
                    for (int x = gt; x < 16 * 16; x += 128) {   // (token i, 16-B chunk c)
                        const int i = x >> 4, c = x & 15;
                        if (tb + i >= n) break;
                        const uint4 val = *reinterpret_cast<const uint4*>(stg + i * 128 + 8 * c);
                        if (isdry) continue;
                        if (d.x == 4) {
                            st_v4_hint(P.shout + static_cast<size_t>(NMAX * p + tb + i) * DIM + r * 128 + 8 * c, val);
                            continue;
                        }
                        const unsigned ev = T.ent[eb + tb + i];
                        const int t = static_cast<int>(ev & 511u);
                        unsigned km = ev >> 9;
                        while (km) {
                            const int k = __ffs(km) - 1;
                            km &= km - 1;
                            if (!(TPB2_PROBE & 1024)) st_v4_hint(P.outp + (static_cast<size_t>(t) * TOPK_MAX + k) * OUTP_LD + r * 128 + 8 * c, val);
                        }
                    }
                    named_bar(1 + g, 128);   // (stg reused by the next row tile)
                    if (pub) xt.rec(XT_END(47));
                    if (DBGP && pub && !isdry) DBGP[b * 32 + 19] += ctime() - te1;
                    continue;
                }
                // ---- FC1 piece of item d.y, stages [d.z, d.w)
                const int uitem = d.y, item = d.x == 1 ? titem(d.y) : titemS(d.y), s0 = d.z, s1 = d.w;   // (unit / this CTA's tile item)
                bool lastm = false;
                // (rr only for shared: measured worse in SHEARLY stream-K, M1-2; routed: rr K-pieces, rS > 1)
                const bool rtail = d.x == 1 && rr && tr > 0 && uitem >= ritem0;
                const bool shm = d.x == 3 && (rr || csr) && SHQ > 1, rtm = d.x == 1 && rr && (rS > 1 || rtail);
                if (d.x == 3 && shh >= 0 && !isdry) {   // ---- SHGU: up CTA -> its tile as (epoch | fp32) words; gate CTA polls them
                    unsigned long long* pq0 = reinterpret_cast<unsigned long long*>(P.part);
                    const unsigned long long tag = static_cast<unsigned long long>(epoch) << 32;
                    if (shh == 1) {
                        unsigned long long* pq = pq0 + static_cast<size_t>(b) * NMAX * 128;
#pragma unroll
                        for (int i = 0; i < 16; ++i) if (tb + i < n) st_relaxed64(pq + (tb + i) * 128 + L, tag | __float_as_uint(vu[i]));
                        continue;
                    }
                    if (pub) xt.rec(XT_BEG(44));
                    const unsigned long long* pc = pq0 + static_cast<size_t>(cta_of(pb + 1)) * NMAX * 128;
                    for (;;) {
                        bool ok = true;
#pragma unroll
                        for (int i = 0; i < 16; ++i)
                            if (tb + i < n) {
                                const unsigned long long w = ld_relaxed64(pc + (tb + i) * 128 + L);
                                ok &= static_cast<unsigned>(w >> 32) == epoch; vu[i] = __uint_as_float(static_cast<unsigned>(w));
                            }
                        if (__all_sync(0xffffffffu, ok)) break;
                        __nanosleep(TPB2_LL_SLEEP);
                    }
                    lastm = true;
                }
                if (TPB2_LLRR && (shm || rtm) && !isdry && M <= TPB2_LLRR_MAXM) {   // ---- LLRR: static owner = piece 0, contributors' LL partials
                    const int npcr = rtail ? tS2 : rS;
                    const int npc = shm ? SHQ : npcr, spc = shm ? SHC : NSTG1 / npcr;   // pieces per item, stages per piece
                    const int qp = s0 / spc, b0u = pb - qp;   // (piece q on unit b0u + q)
                    const int lslot = rtail ? MAXG : 0;
                    unsigned long long* pq0 = reinterpret_cast<unsigned long long*>(P.part);
                    const unsigned long long tag = static_cast<unsigned long long>(epoch) << 32;
                    if (qp > 0) {   // contributor: [token][row] gate (shared: up at token 8 +) as (epoch | fp32) words in its own slot
                        if (pub) xt.rec(XT_BEG(41), xd0(d), xd1(d));
                        unsigned long long* pq = pq0 + static_cast<size_t>(lslot + b) * NMAX * 128;
#pragma unroll
                        for (int i = 0; i < 16; ++i)
                            if (tb + i < n) {
                                st_relaxed64(pq + (tb + i) * 128 + L, tag | __float_as_uint(v[i]));
                                if (shm) st_relaxed64(pq + (8 + tb + i) * 128 + L, tag | __float_as_uint(vu[i]));
                            }
                        if (pub) xt.rec(XT_END(41));
                        continue;
                    }
                    if (pub) xt.rec(XT_BEG(44));
                    for (int c = 1; c < npc; ++c) {   // owner: pieces 1 .. npc - 1 polled and added in piece order (== LASTM's order)
                        const unsigned long long* pc = pq0 + static_cast<size_t>(lslot + cta_of(b0u + c)) * NMAX * 128;
                        float ag[16], au[16];
                        for (;;) {
                            bool ok = true;
#pragma unroll
                            for (int i = 0; i < 16; ++i) {
                                ag[i] = 0.f; au[i] = 0.f;
                                if (tb + i < n) {
                                    const unsigned long long w = ld_relaxed64(pc + (tb + i) * 128 + L);
                                    ok &= static_cast<unsigned>(w >> 32) == epoch; ag[i] = __uint_as_float(static_cast<unsigned>(w));
                                    if (shm) {
                                        const unsigned long long wu = ld_relaxed64(pc + (8 + tb + i) * 128 + L);
                                        ok &= static_cast<unsigned>(wu >> 32) == epoch; au[i] = __uint_as_float(static_cast<unsigned>(wu));
                                    }
                                }
                            }
                            if (__all_sync(0xffffffffu, ok)) break;
                            __nanosleep(TPB2_LL_SLEEP);
                        }
#pragma unroll
                        for (int i = 0; i < 16; ++i) if (tb + i < n) { v[i] += ag[i]; if (shm) vu[i] += au[i]; }
                    }
                    lastm = true;
                }
                if (TPB2_LASTM && (shm || rtm) && !isdry && !lastm) {
                    const int npcr = rtail ? tS2 : rS;
                    const int npc = shm ? SHQ : npcr, spc = shm ? SHC : NSTG1 / npcr;   // pieces per item, stages per piece
                    if (pub) xt.rec(XT_BEG(41), xd0(d), xd1(d));
                    const int lslot = rtail ? MAXG : 0;   // (tail pieces: own slots -- the unit's earlier shared piece may still be read)
                    float* pp = P.part + static_cast<size_t>(lslot + b) * 2 * NMAX * 128;
                    if (TPB2_LMV) {   // (row-major partials: a thread's 16 token columns contiguous -> 16-B stores / loads, small code)
                        float4* g4 = reinterpret_cast<float4*>(pp + L * NMAX + tb);
                        float4* u4 = reinterpret_cast<float4*>(pp + NMAX * 128 + L * NMAX + tb);
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            if (tb + 4 * j < npad) {
                                g4[j] = make_float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
                                if (shm) u4[j] = make_float4(vu[4 * j], vu[4 * j + 1], vu[4 * j + 2], vu[4 * j + 3]);
                            }
                    } else {
#pragma unroll
                    for (int i = 0; i < 16; ++i) if (tb + i < npad) { pp[(tb + i) * 128 + L] = v[i]; if (shm) pp[(NMAX + tb + i) * 128 + L] = vu[i]; }
                    }
                    named_bar(3, 256);
                    if (pub) {
                        xt.rec(XT_END(41)); xt.rec(XT_BEG(48));
                        unsigned* cnt = P.tcnt + par * MAXM * NRT + (shm ? item : 64 + (rtail ? uitem - ritem0 : item));   // (zeroed by the previous launch)
                        T.misc[22] = atom_add_acq_rel(cnt, 1u) == static_cast<unsigned>(npc - 1);
                        xt.rec(XT_END(48));
                    }
                    named_bar(3, 256);
                    if (!*reinterpret_cast<volatile int*>(T.misc + 22)) continue;
                    lastm = true;
                    if (pub) xt.rec(XT_BEG(44));
                    const int b0u = pb - s0 / spc;   // piece q on unit b0u + q (this rank's CTA of it)
                    constexpr int MPC = TPB2_SHQ > 4 ? TPB2_SHQ : 4;
                    if (TPB2_LMV) {   // every piece's 16 token columns of gate (and up) in flight together, then summed in piece order
                        const int nj = npad <= 8 ? 2 : 4;   // (float4 groups of this thread's columns)
                        float4 qg[MPC][4], qu[MPC][4];
#pragma unroll
                        for (int c = 0; c < MPC; ++c) {
                            const float* pc = P.part + static_cast<size_t>(lslot + cta_of(b0u + c)) * 2 * NMAX * 128 + L * NMAX + tb;
                            const bool ld = c < npc && b0u + c != pb;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                qg[c][j] = (ld && j < nj) ? __ldcg(reinterpret_cast<const float4*>(pc) + j) : make_float4(0.f, 0.f, 0.f, 0.f);
                                qu[c][j] = (ld && j < nj && shm) ? __ldcg(reinterpret_cast<const float4*>(pc + NMAX * 128) + j) : make_float4(0.f, 0.f, 0.f, 0.f);
                            }
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            float g[4] = {0.f, 0.f, 0.f, 0.f}, u[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                            for (int c = 0; c < MPC; ++c) {
                                if (c >= npc) break;
                                const bool own = b0u + c == pb;
                                const float a4[4] = {qg[c][j].x, qg[c][j].y, qg[c][j].z, qg[c][j].w}, b4[4] = {qu[c][j].x, qu[c][j].y, qu[c][j].z, qu[c][j].w};
#pragma unroll
                                for (int e = 0; e < 4; ++e) {
                                    const float gv = own ? v[4 * j + e] : a4[e], uv = own ? vu[4 * j + e] : b4[e];
                                    g[e] = c == 0 ? gv : g[e] + gv; u[e] = c == 0 ? uv : u[e] + uv;
                                }
                            }
#pragma unroll
                            for (int e = 0; e < 4; ++e) { v[4 * j + e] = g[e]; vu[4 * j + e] = u[e]; }
                        }
                    } else if (npad <= 8) {   // small panels: gate and up of all pieces in one round trip (8 token columns)
                        float qg[MPC][8], qu[MPC][8];
#pragma unroll
                        for (int c = 0; c < MPC; ++c) {
                            const float* pc = P.part + static_cast<size_t>(lslot + cta_of(b0u + c)) * 2 * NMAX * 128;
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const bool own = b0u + c == pb, ok = c < npc && tb + i < npad;
                                qg[c][i] = own ? v[i] : (ok ? __ldcg(pc + (tb + i) * 128 + L) : 0.f);
                                qu[c][i] = own ? vu[i] : ((ok && shm) ? __ldcg(pc + (NMAX + tb + i) * 128 + L) : 0.f);
                            }
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            float sg = qg[0][i], su = qu[0][i];
#pragma unroll
                            for (int c = 1; c < MPC; ++c) if (c < npc) { sg += qg[c][i]; su += qu[c][i]; }
                            v[i] = sg; vu[i] = su;
                        }
                    } else {
#pragma unroll 1
                    for (int h = 0; h < (shm ? 2 : 1); ++h) {   // gate, then up: the pieces' values loaded together, summed in piece order
                        float qv[MPC][16];
#pragma unroll
                        for (int c = 0; c < MPC; ++c) {
                            const float* pc = P.part + static_cast<size_t>(lslot + cta_of(b0u + c)) * 2 * NMAX * 128 + (h ? NMAX * 128 : 0);
#pragma unroll
                            for (int i = 0; i < 16; ++i)
                                qv[c][i] = (b0u + c == pb) ? (h ? vu[i] : v[i]) : ((c < npc && tb + i < npad) ? __ldcg(pc + (tb + i) * 128 + L) : 0.f);
                        }
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            float sum = qv[0][i];
#pragma unroll
                            for (int c = 1; c < MPC; ++c) if (c < npc) sum += qv[c][i];
                            if (h) vu[i] = sum; else v[i] = sum;
                        }
                    }
                    }
                }
                if (CS && csr && d.x == 1) {   // CSPLIT: the item's K-halves on the cluster's CTAs (rank 1: partial -> rank 0's smem)
                    const int slot = csk & 1;
                    const uint32_t mbuf = sb + OFF_TAB + TAB_MBUF + slot * NMAX * 128 * 4;
                    if (crank == 1u) {
                        if (pub) xt.rec(XT_BEG(41), xd0(d), xd1(d));
                        if (csk >= 2) mbar_wait(bar(B_MEMPTY + slot), static_cast<uint32_t>((csk >> 1) - 1) & 1u);   // (rank 0 read the slot's last one)
                        const uint32_t pm = peer_addr(mbuf, 0u);
#pragma unroll
                        for (int i = 0; i < 16; ++i) if (tb + i < npad) st_cluster_f32(pm + static_cast<uint32_t>(((tb + i) * 128 + L) * 4), v[i]);
                        named_bar(3, 256);
                        if (pub) { mbar_arrive_remote(peer_addr(bar(B_MFULL + slot), 0u)); xt.rec(XT_END(41)); }   // (release.cluster: the 256 threads' stores)
                        ++csk;
                        continue;
                    }
                    if (pub) xt.rec(XT_BEG(43), 1u);
                    mbar_wait_cl(bar(B_MFULL + slot), static_cast<uint32_t>(csk >> 1) & 1u);
                    const float* mb = reinterpret_cast<const float*>(smem + OFF_TAB + TAB_MBUF) + slot * NMAX * 128;
#pragma unroll
                    for (int i = 0; i < 16; ++i) if (tb + i < npad) v[i] += mb[(tb + i) * 128 + L];
                    named_bar(3, 256);
                    if (pub) { mbar_arrive_remote_relaxed(peer_addr(bar(B_MEMPTY + slot), 1u)); xt.rec(XT_END(43)); xt.rec(XT_BEG(44)); }
                    ++csk;
                    lastm = true;   // (merged: skip the global contributor / owner paths)
                }
                if (!WHOLE && !lastm && s0 > 0 && TPB2_LLMERGE) {   // non-owner: fp32 partial with the epoch tag -> part[b] (no flag, no barrier)
                    unsigned long long* pq = reinterpret_cast<unsigned long long*>(P.part) + static_cast<size_t>(b) * 2 * NMAX * 128;
                    const unsigned long long tag = static_cast<unsigned long long>(epoch) << 32;
#pragma unroll
                    for (int i = 0; i < 16; ++i) if (tb + i < npad) st_relaxed64(pq + (tb + i) * 128 + L, tag | __float_as_uint(v[i]));
                    if (d.x == 3) {
#pragma unroll
                        for (int i = 0; i < 16; ++i) if (tb + i < npad) st_relaxed64(pq + (NMAX + tb + i) * 128 + L, tag | __float_as_uint(vu[i]));
                    }
                    continue;
                }
                const int pslot = (rr && d.x == 1) ? MAXG : 0;   // (rr: routed split pieces are tail pieces: their own partial slots)
                if (!WHOLE && !lastm && s0 > 0) {   // non-owner (this CTA's first piece): fp32 partial -> part[b], flag
                    if (pub) xt.rec(XT_BEG(41), xd0(d), xd1(d));
                    float* pp = P.part + static_cast<size_t>(pslot + b) * 2 * NMAX * 128;
#pragma unroll
                    for (int i = 0; i < 16; ++i) if (tb + i < npad) pp[(tb + i) * 128 + L] = v[i];
                    if (d.x == 3) {
#pragma unroll
                        for (int i = 0; i < 16; ++i) if (tb + i < npad) pp[(NMAX + tb + i) * 128 + L] = vu[i];
                    }
                    named_bar(3, 256);
                    if (pub) { xt.rec(XT_END(41)); xt.rec(XT_BEG(48)); }
                    if (pub) { fence_acq_rel(); st_release(P.part_flags + pslot + b, epoch); }
                    if (pub) xt.rec(XT_END(48));
                    continue;
                }
                if (DBGP && pub) DBGP[b * 32 + 12] = gtime();
                if (!WHOLE && !lastm && s1 < (d.x == 1 ? NSTG1 : NSTG1S)) {   // owner of a split item: add the partials of the CTAs b + 1, .. covering its other stages, in CTA order
                    const int iend = d.x == 1 ? (uitem + 1) * NSTG1 : R1 + (uitem + 1) * NSTG1S;   // (unified stage index, units)
                    int cend = pb + 1;   // (contributor units pb + 1 .. cend - 1; this rank's CTAs of them)
                    if (rr) cend = pb + (d.x == 1 ? tS : SHQ);   // (rr: shared items in SHQ pieces on units pb .. pb + SHQ - 1)
                    else if (skq && d.x == 3) cend = pb + SHQ;
                    else while (cend < PG && ustart(cend) < iend) ++cend;
                    const unsigned long long tmw = (DBGP && pub) ? ctime() : 0ull;
                    if (pub) xt.rec(XT_BEG(43), static_cast<uint32_t>(cend - pb - 1));
                    if (TPB2_LLMERGE) {   // contributors in CTA order: poll each one's tagged partial until every value carries this epoch
                        const unsigned long long* pq0 = reinterpret_cast<const unsigned long long*>(P.part);
#pragma unroll 1
                        for (int c = pb + 1; c < cend; ++c) {
                            const unsigned long long* pq = pq0 + static_cast<size_t>(cta_of(c)) * 2 * NMAX * 128;
                            unsigned long long qv[16], qu[16];
                            for (;;) {
                                bool ok = true;
#pragma unroll
                                for (int i = 0; i < 16; ++i) {
                                    qv[i] = tb + i < npad ? ld_relaxed64(pq + (tb + i) * 128 + L) : (static_cast<unsigned long long>(epoch) << 32);
                                    qu[i] = (d.x == 3 && tb + i < npad) ? ld_relaxed64(pq + (NMAX + tb + i) * 128 + L) : (static_cast<unsigned long long>(epoch) << 32);
                                }
#pragma unroll
                                for (int i = 0; i < 16; ++i) ok &= static_cast<unsigned>(qv[i] >> 32) == epoch && static_cast<unsigned>(qu[i] >> 32) == epoch;
                                if (__all_sync(0xffffffffu, ok)) break;
                                __nanosleep(TPB2_LL_SLEEP);   // (back off: the contributors are still computing)
                            }
#pragma unroll
                            for (int i = 0; i < 16; ++i) {
                                if (tb + i < npad) {
                                    v[i] += __uint_as_float(static_cast<unsigned>(qv[i]));
                                    if (d.x == 3) vu[i] += __uint_as_float(static_cast<unsigned>(qu[i]));
                                }
                            }
                        }
                        if (pub) { xt.rec(XT_END(43)); xt.rec(XT_BEG(44)); }
                    } else {
                    for (int c = pb + 1 + lane; c < cend; c += 32) wait_eq(P.part_flags + pslot + cta_of(c), epoch);
                    __syncwarp();
                    if (pub) { xt.rec(XT_END(43)); xt.rec(XT_BEG(44)); }
                    }
                    if (DBGP && pub) DBGP[b * 32 + 17] += ctime() - tmw;   // (merge: cycles waiting for the contributors' flags)
                    if (TPB2_LLMERGE) {
                    } else if (d.x == 1) {
                        constexpr int MB = TPB2_MERGE_B;
                        for (int c0 = pb + 1; c0 < cend; c0 += MB) {   // MB contributors' partials loaded together, added in CTA order
                            float pv[MB][16];
#pragma unroll
                            for (int j = 0; j < MB; ++j) {
                                const float* pp = P.part + static_cast<size_t>(pslot + cta_of(c0 + j)) * 2 * NMAX * 128;
#pragma unroll
                                for (int i = 0; i < 16; ++i) pv[j][i] = (c0 + j < cend && tb + i < npad) ? __ldcg(pp + (tb + i) * 128 + L) : 0.f;
                            }
#pragma unroll
                            for (int j = 0; j < MB; ++j)
#pragma unroll
                                for (int i = 0; i < 16; ++i) if (c0 + j < cend) v[i] += pv[j][i];
                        }
                    } else {
                        constexpr int MB = TPB2_MERGE_BS;   // (shared items: up to ~10 contributors at small M -- batched round trips)
                        for (int c0 = pb + 1; c0 < cend; c0 += MB) {
                            float pv[MB][16], pu[MB][16];
#pragma unroll
                            for (int j = 0; j < MB; ++j) {
                                const float* pp = P.part + static_cast<size_t>(cta_of(c0 + j)) * 2 * NMAX * 128;
#pragma unroll
                                for (int i = 0; i < 16; ++i) {
                                    const bool ok = c0 + j < cend && tb + i < npad;
                                    pv[j][i] = ok ? __ldcg(pp + (tb + i) * 128 + L) : 0.f;
                                    pu[j][i] = ok ? __ldcg(pp + (NMAX + tb + i) * 128 + L) : 0.f;
                                }
                            }
#pragma unroll
                            for (int j = 0; j < MB; ++j)
#pragma unroll
                                for (int i = 0; i < 16; ++i) if (c0 + j < cend) { v[i] += pv[j][i]; vu[i] += pu[j][i]; }
                        }
                    }
                }
                if (DBGP && pub && !isdry) DBGP[b * 32 + 13] = gtime();
                if (pub) {
                    if (lastm || s1 < (d.x == 1 ? NSTG1 : NSTG1S)) xt.rec(XT_END(44));
                    xt.rec(XT_BEG(45), xd0(d), xd1(d));
                }
                if ((TPB2_DLY_SITE == 3 && d.x == 1) || (TPB2_DLY_SITE == 5 && d.x == 3)) dly_spin(TPB2_DLY_NS);
                const unsigned long long t_sw0 = (DBGP && pub) ? gtime() : 0ull;
                if (d.x == 3) {
                    // ---- shared FC1 item (128 h columns = lanes): DeepGEMM bf16 gate / up, silu_and_mul_clamp (bf16 clamp, fp32 silu * up, bf16),
                    // per-128 FP8 (ceil rule, eps 1e-10: sglang per_token_group_quant ue8m0) -> the shared h image + SFB (4 equal bytes)
                    const int ng = min(TPB2_ACTU_ON ? min(16, TPB2_MMAX) : 16, n - tb), kq = item % NT1S;
                    float am_mine = 0.f;
                    float r[16];   // (TPB2_ROLL layout: unprocessed gate values at the front, finished activations appended at the back)
#pragma unroll
                    for (int i = 0; i < 16; ++i) r[i] = v[i];
#if TPB2_ACTU_ON
#pragma unroll
                    for (int i = 0; i < 16; ++i) {   // (r[i] = token i's activation, as the ROLL rotation leaves it)
                        if (i >= ng) break;
                        const float gq = fminf(bf16r(r[i]), 10.0f), uq = fminf(fmaxf(bf16r(vu[i]), -10.0f), 10.0f);
                        const float a2 = bf16r((gq / (1.0f + __expf(-gq))) * uq);
                        float m = fabsf(a2);
#pragma unroll
                        for (int o = 1; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                        am_mine = lane == i ? m : am_mine;
                        r[i] = a2;
                    }
#else
#pragma unroll 1
                    for (int i = 0; i < ng; ++i) {
                        const float gq = fminf(bf16r(r[0]), 10.0f), uq = fminf(fmaxf(bf16r(vu[0]), -10.0f), 10.0f);
                        const float a2 = bf16r((gq / (1.0f + __expf(-gq))) * uq);
                        float m = fabsf(a2);
#pragma unroll
                        for (int o = 1; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                        am_mine = lane == i ? m : am_mine;
#pragma unroll
                        for (int k = 0; k < 15; ++k) { r[k] = r[k + 1]; vu[k] = vu[k + 1]; }
                        r[15] = a2;
                    }
#pragma unroll 1
                    for (int k2 = ng; k2 < 16; ++k2) {   // activations to the front (token i at r[i])
                        const float t0 = r[0];
#pragma unroll
                        for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
                        r[15] = t0;
                    }
#endif
                    if (lane < 16 && tb + lane < NMAX) T.am[q * NMAX + tb + lane] = am_mine;
                    named_bar(1 + g, 128);
                    uint8_t* hb = P.hsimg + (static_cast<size_t>(p) * NCH2 + kq) * NMAX * 128;
#if TPB2_ACTU_ON
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        if (i >= ng) break;
                        const float a = r[i];
#else
#pragma unroll 1
                    for (int i = 0; i < ng; ++i) {
                        const float a = r[0];
#pragma unroll
                        for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
#endif
                        const int tok = tb + i;
                        const float amax = fmaxf(fmaxf(T.am[tok], T.am[NMAX + tok]), fmaxf(T.am[2 * NMAX + tok], T.am[3 * NMAX + tok]));
                        const int ex = ue8m0_exp(fmaxf(amax, 1e-10f));
                        const uint32_t byte = e4m3_of(a * exp2i(-ex));
                        uint32_t word = 0;   // column L = 32 q + lane: granule 2 q + lane / 16, byte lane % 16; lanes (lane & 15) < 4 gather word lane & 3
#pragma unroll
                        for (int bb = 0; bb < 4; ++bb) word |= (__shfl_sync(0xffffffffu, byte, (lane & 16) + 4 * (lane & 3) + bb) & 0xFFu) << (8 * bb);
                        if ((lane & 15) < 4)
                            *reinterpret_cast<uint32_t*>(hb + (tok >> 3) * 1024 + (tok & 7) * 128 + (((2 * q + (lane >> 4)) ^ (tok & 7)) << 4) + 4 * (lane & 3)) = word;
                        if (q == 0 && lane == 0)
                            *reinterpret_cast<uint32_t*>(P.hssf + (static_cast<size_t>(p) * NSTG2 + (kq >> 1)) * 512 + tok * 16 + 8 * (kq & 1)) =
                                static_cast<uint32_t>(ex + 127) * 0x01010101u;
                    }
                    named_bar(3, 256);
                    if (pub) { xt.rec(XT_END(45)); xt.rec(XT_BEG(46)); }
                    if (pub && TPB2_HPUB && TPB2_RTPUB && !TPB2_T1W) { hpub_enqueue(smem, Ptot + p, true); xt.rec(XT_END(46), static_cast<uint32_t>(Ptot + p), 2u); }
                    else if (pub && TPB2_PUBRED && (f2s || fc2l)) { red_release_add(P.pan_cnt + par * MAXP + Ptot + p, 1u); xt.rec(XT_END(46), static_cast<uint32_t>(Ptot + p), 2u); }
                    else if (pub) {
                        const bool lastit = ((TPB2_PROBE & 256) ? atomicAdd(P.pan_cnt + par * MAXP + Ptot + p, 1u) : atom_add_acq_rel(P.pan_cnt + par * MAXP + Ptot + p, 1u)) == NT1S - 1u;
                        if (lastit) {   // the panel's last FC1 item: append it
                            unsigned* rl = P.rdy + par * (MAXP + 32);
                            st_release(rl + atomicAdd(rl + MAXP, 1u), static_cast<unsigned>(Ptot + p + 1));
                        }
                        xt.rec(XT_END(46), static_cast<uint32_t>(Ptot + p), (lastit ? 1u : 0u) | 2u);
                    }
                    continue;
                }
                // SwiGLU: the gate of an up lane's h column sits in lane ^ 8 of the same warp.  TPB2_ROLL: compact loops over the live
                // tokens on a rotating register copy (the code runs i-cache cold on the critical path: its size, not its count, costs)
                const int ng = min(TPB2_ACTU_ON ? min(16, TPB2_MMAX) : 16, n - tb);   // this group's live tokens (warp-uniform; <= 0: none)
                float am_mine = 0.f;   // lane i: the warp's amax of token tb + i
                const int tile = item % NT1, q2 = tile >> 1;   // FC2 k-chunk of these 64 h columns: 64 (tile & 1) + 16 q + ..
                uint8_t* himg_p = P.himg + static_cast<size_t>(p) * NCH2 * NMAX * 128 + static_cast<size_t>(q2) * NMAX * 128;
                const int gran = 4 * (tile & 1) + q;   // 16-B granule of this warp's 16 columns in the 128-B row
#if TPB2_ROLL && (TPB2_ACTX || TPB2_ACTX_MINM)
                float r[16];   // unprocessed values at the front, finished activations appended at the back
#pragma unroll
                for (int i = 0; i < 16; ++i) r[i] = v[i];
                const bool actx = TPB2_ACTX || M >= TPB2_ACTX_MINM;   // (4 tokens per round: wins with wide panels)
                if (actx) {
                const int nr4 = (ng + 3) >> 2;
#pragma unroll 1
                for (int i0 = 0; i0 < 4 * nr4; i0 += 4) {
                    float a4[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float x = r[j];
                        const float gv = __shfl_xor_sync(0xffffffffu, x, 8);
                        float a2 = 0.f;
                        if (up_lane) {
                            const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(x, -10.0f), 10.0f);
                            a2 = silu_mul(gg, uu);
                        }
                        a4[j] = a2;
                    }
                    uint32_t m4[4];   // (amax over the warp's lanes: gate lanes hold 0; |a| >= 0 -> max of the bit patterns == fmaxf)
#pragma unroll
                    for (int j = 0; j < 4; ++j) m4[j] = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(a4[j])));
#pragma unroll
                    for (int j = 0; j < 4; ++j) am_mine = lane == i0 + j ? __uint_as_float(m4[j]) : am_mine;
#pragma unroll
                    for (int k = 0; k < 12; ++k) r[k] = r[k + 4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) r[12 + j] = a4[j];
                }
#pragma unroll 1
                for (int k2 = nr4; k2 < 4; ++k2) {   // rotate the activations to the front (token i at r[i])
                    float t4[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) t4[j] = r[j];
#pragma unroll
                    for (int k = 0; k < 12; ++k) r[k] = r[k + 4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) r[12 + j] = t4[j];
                }
                } else {
#pragma unroll 1
                for (int i = 0; i < ng; ++i) {
                    const float x = r[0];
                    const float gv = __shfl_xor_sync(0xffffffffu, x, 8);
                    float a2 = 0.f;
                    if (up_lane) {
                        const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(x, -10.0f), 10.0f);
                        a2 = silu_mul(gg, uu);
                    }
                    float m = fabsf(a2);   // amax over the warp's 16 up lanes (gate lanes hold 0)
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
                    am_mine = lane == i ? m : am_mine;
#pragma unroll
                    for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
                    r[15] = a2;
                }
#pragma unroll 1
                for (int k2 = ng; k2 < 16; ++k2) {   // rotate the activations to the front (token i at r[i])
                    const float t0 = r[0];
#pragma unroll
                    for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
                    r[15] = t0;
                }
                }
#elif TPB2_ROLL && !TPB2_ACTU_ON
                float r[16];   // unprocessed values at the front, finished activations appended at the back
#pragma unroll
                for (int i = 0; i < 16; ++i) r[i] = v[i];
#pragma unroll 1
                for (int i = 0; i < ng; ++i) {
                    const float x = r[0];
                    const float gv = __shfl_xor_sync(0xffffffffu, x, 8);
                    float a2 = 0.f;
                    if (up_lane) {
                        const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(x, -10.0f), 10.0f);
                        a2 = silu_mul(gg, uu);
                    }
                    float m = fabsf(a2);   // amax over the warp's 16 up lanes (gate lanes hold 0)
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
                    am_mine = lane == i ? m : am_mine;
#pragma unroll
                    for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
                    r[15] = a2;
                }
#pragma unroll 1
                for (int k2 = ng; k2 < 16; ++k2) {   // rotate the activations to the front (token i at r[i])
                    const float t0 = r[0];
#pragma unroll
                    for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
                    r[15] = t0;
                }
#else
                float act[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    act[i] = 0.f;
                    if (i < ng) {
                        const float gv = __shfl_xor_sync(0xffffffffu, v[i], 8);
                        float a2 = 0.f;
                        if (up_lane) {
                            const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(v[i], -10.0f), 10.0f);
                            a2 = silu_mul(gg, uu);
                        }
                        act[i] = a2;
                        float m = fabsf(a2);   // amax over the warp's 16 up lanes (gate lanes hold 0)
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));    // (every lane gets it: lane i holds token i's)
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
                        am_mine = lane == i ? m : am_mine;
                    }
                }
#endif
                if (DBGP && pub && !isdry) DBGP[b * 32 + 14] = gtime();
                if (pub) xt.rec(XT_PT(73));
                if (lane < 16 && tb + lane < NMAX) T.am[q * NMAX + tb + lane] = am_mine;   // 32-column scale blocks = warp pairs (q, q ^ 1)
                named_bar(1 + g, 128);
                if (pub) xt.rec(XT_PT(74));
                if (DBGP && pub) DBGP[b * 32 + 15] = gtime();
#if TPB2_ROLL && (TPB2_ACTX || TPB2_ACTX_MINM)
                if (actx) {
#pragma unroll 1
                for (int i0 = 0; i0 < ng; i0 += 4) {   // 4 tokens per round (tokens tb + i0 + j <= 31: T.am rows hold NMAX)
                    uint32_t w4[4];
                    int ex4[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int tok = tb + i0 + j;
                        const float amax = fmaxf(T.am[q * NMAX + tok], T.am[(q ^ 1) * NMAX + tok]);
                        ex4[j] = ue8m0_exp_floor(amax);
                        const uint32_t byte = up_lane ? e4m3_of(r[j] * exp2i(-ex4[j])) : 0u;
                        uint32_t word = 0;
#pragma unroll
                        for (int bb = 0; bb < 4; ++bb) {
                            const int jb = 4 * (lane & 3) + bb;
                            word |= (__shfl_sync(0xffffffffu, byte, ((jb >> 1) & 7) | ((jb & 1) << 4)) & 0xFFu) << (8 * bb);
                        }
                        w4[j] = word;
                    }
#pragma unroll
                    for (int k = 0; k < 12; ++k) r[k] = r[k + 4];
                    if (isdry) continue;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int tok = tb + i0 + j;
                        if (i0 + j < ng) {
                            if (lane < 4)
                                *reinterpret_cast<uint32_t*>(himg_p + (tok >> 3) * 1024 + (tok & 7) * 128 + ((gran ^ (tok & 7)) << 4) + 4 * lane) = w4[j];
                            if (lane == 0 && (q & 1) == 0) {
                                const int kb = 2 * (tile & 1) + (q >> 1);
                                P.hsf[(static_cast<size_t>(p) * NSTG2 + (q2 >> 1)) * 512 + tok * 16 + 8 * (q2 & 1) + kb] = static_cast<uint8_t>(ex4[j] + 127);
                            }
                        }
                    }
                }
                } else {
#endif
#if TPB2_ROLL && !TPB2_ACTU_ON
#pragma unroll 1
                for (int i = 0; i < ng; ++i) {
                    const float a = r[0];
#pragma unroll
                    for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
#else
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (i >= ng) break;
                    const float a = act[i];
#endif
                    const int tok = tb + i;
                    const float amax = fmaxf(T.am[q * NMAX + tok], T.am[(q ^ 1) * NMAX + tok]);
                    const int ex = ue8m0_exp_floor(amax);
                    const uint32_t byte = up_lane ? e4m3_of(a * exp2i(-ex)) : 0u;
                    uint32_t word = 0;   // lane wd (< 4) gathers bytes 4 wd .. 4 wd + 3 of the granule (byte jb is in up lane ((jb >> 1) & 7) | (jb & 1) << 4)
#pragma unroll
                    for (int bb = 0; bb < 4; ++bb) {
                        const int jb = 4 * (lane & 3) + bb;
                        word |= (__shfl_sync(0xffffffffu, byte, ((jb >> 1) & 7) | ((jb & 1) << 4)) & 0xFFu) << (8 * bb);
                    }
                    if (isdry) continue;
                    if (lane < 4)
                        *reinterpret_cast<uint32_t*>(himg_p + (tok >> 3) * 1024 + (tok & 7) * 128 + ((gran ^ (tok & 7)) << 4) + 4 * lane) = word;
                    if (lane == 0 && (q & 1) == 0) {   // one scale byte per (token, 32-column block)
                        const int kb = 2 * (tile & 1) + (q >> 1);
                        P.hsf[(static_cast<size_t>(p) * NSTG2 + (q2 >> 1)) * 512 + tok * 16 + 8 * (q2 & 1) + kb] = static_cast<uint8_t>(ex + 127);
                    }
                }
#if TPB2_ROLL && (TPB2_ACTX || TPB2_ACTX_MINM)
                }   // (ACTX / ROLL quantization)
#endif
                if (pub) xt.rec(XT_PT(75));
                named_bar(3, 256);
                if (pub) { xt.rec(XT_END(45)); xt.rec(XT_BEG(46)); }
                if (pub && !isdry && TPB2_HPUB && TPB2_RTPUB && !TPB2_T1W) { hpub_enqueue(smem, p, false); xt.rec(XT_END(46), static_cast<uint32_t>(p), 0u); }
                else if (pub && !isdry && TPB2_PUBRED && (f2s || fc2l)) { red_release_add(P.pan_cnt + par * MAXP + p, 1u); xt.rec(XT_END(46), static_cast<uint32_t>(p), 0u); }
                else if (pub && !isdry) {
                    const bool lastit = ((TPB2_PROBE & 256) ? atomicAdd(P.pan_cnt + par * MAXP + p, 1u) : atom_add_acq_rel(P.pan_cnt + par * MAXP + p, 1u)) == NT1 - 1u;
                    if (lastit) {   // the panel's last FC1 item: append it to the ready list
                        unsigned* rl = P.rdy + par * (MAXP + 32);
                        st_release(rl + atomicAdd(rl + MAXP, 1u), static_cast<unsigned>(p + 1));
                    }
                    xt.rec(XT_END(46), static_cast<uint32_t>(p), lastit ? 1u : 0u);
                    if (DBGP) DBGP[b * 32 + 4] = gtime();
                }
            }
            if (!fc1 && TPB2_T1W && !isdry) {   // FC2 unit done (both groups): hand it to the T1 warp (it counts the unit's (token, 256-row tile)
                                      // contributions with a gpu-scope release: no fence on the epilogue's critical path)
                named_bar(3, 256);
                if (pub) {
                    volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
                    const int qn = q[T1Q];
                    while (qn - q[T1Q + 2] >= T1Q) {}   // (queue full: the T1 warp's consumed count)
                    q[qn % T1Q] = (d.x << 24) | d.y;
                    __threadfence_block();
                    q[T1Q] = qn + 1;
                }
            }
            if (!fc1 && !TPB2_T1W && TPB2_RTPUB && !isdry && !t1ll_on(P)) {   // FC2 unit done (both groups): queue it for the warp-7 publisher
                named_bar(3, 256);
                if (pub) {
                    volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
                    const int qn = q[T1Q];
                    while (qn - q[T1Q + 2] >= T1Q) {}   // (queue full)
                    q[qn % T1Q] = TPB2_FC2VAR ? (d.y | (d.z << 24)) : d.y;
                    __threadfence_block();
                    q[T1Q] = qn + 1;
                }
            }
            if (!fc1 && !TPB2_T1W && !TPB2_RTPUB) {   // FC2 unit done (both groups): count its row tiles toward T1
                named_bar(3, 256);
                if (pub && !isdry) {
                    const int r0 = f2r0(d), cn = f2n(d);
                    int ij[8];   // T1 tile i = FC2 row tiles 2 i, 2 i + 1 (constant indices: rt_n stays in registers)
#pragma unroll
                    for (int j = 0; j < 8; ++j) ij[j] = j < cn ? (trow(r0 + j) >> 1) : -1;
#pragma unroll
                    for (int i = 0; i < NRT; ++i) {
                        unsigned a = 0u;
#pragma unroll
                        for (int j = 0; j < 8; ++j) a += ij[j] == i ? 1u : 0u;
                        rt_n[i] += a;
                    }
                }
            }
        }
    }
    // ---- all roles done: TMEM free, then T1 + all-reduce with every thread
    tc_fence_before();
    if (PR || CS) cluster_sync_all(); else __syncthreads();   // (PAIR / CSPLIT: the pair's MMAs / remote traffic all done before either frees anything)
    if (warp == 1) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc." TPB2_CG ".sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(TMEM_COLS) : "memory");
    }
    if (DBGP && tid == 0) DBGP[b * 32 + 5] = gtime();
    if (TPB2_DLY_SITE == 2) dly_spin(TPB2_DLY_NS);
    if (tid == 0) xt.rec(XT_PT(50));
    tail(P, T, small, tid, epoch, par, smem, xt);
    if (tid == 0) xt.rec(XT_END(55));
    __syncthreads();
    if (DBGP && tid == 0) { DBGP[b * 32 + 7] = gtime(); DBGP[b * 32 + 31] = clock64(); }
    if (tid == 0) xt_anchor(TPB2_TRACE ? static_cast<void*>(P.dbg_) : nullptr, true);
    if (tid == 0 && (TPB2_ARV & 32)) {   // (fire-and-forget: the next call reads the count after its griddepcontrol.wait)
        asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(reinterpret_cast<unsigned long long*>(P.work + 2)) : "memory");
    } else if (tid == 0) {
        if (!(TPB2_ARV & 4)) fence_acq_rel();   // (ARV 4: nothing in this kernel reads the counts; the next call is stream-ordered)
        if (atomicAdd(P.work + 1 + par, 1u) == static_cast<unsigned>(G) - 1u) atomicAdd(P.work, 1u);   // last CTA out: next epoch
    }
}

// ------------------------------------------------------------------------------------------------ host
#ifndef B2_NO_TORCH
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                              const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode_fn() {
    static EncodeFn fn = nullptr;
    if (!fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        TORCH_CHECK(q == cudaDriverEntryPointSuccess && p, "cuTensorMapEncodeTiled not found");
        fn = reinterpret_cast<EncodeFn>(p);
    }
    return fn;
}
static void encode_weight_map(CUtensorMap* map, void* ptr, long long rows, int row_bytes, int chunks) {
    const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(chunks)};
    const cuuint64_t strides[2] = {static_cast<cuuint64_t>(row_bytes), 64};
    const cuuint32_t box[3] = {128, 128, KC}, es[3] = {1, 1, 1};
    const CUresult r = encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
}
// both weight maps as 256 bytes of a CPU uint8 tensor (cached by the caller per layer)
torch::Tensor encode_maps(torch::Tensor w13, torch::Tensor w2) {
    TORCH_CHECK(w13.is_cuda() && w2.is_cuda() && w13.element_size() == 1 && w2.element_size() == 1 && w13.is_contiguous() && w2.is_contiguous());
    TORCH_CHECK(w13.dim() == 3 && w13.size(0) == NEXP && w13.size(1) == 2 * INTER && w13.size(2) == DIM / 2, "w13 [256, 1024, 2048]");
    TORCH_CHECK(w2.dim() == 3 && w2.size(0) == NEXP && w2.size(1) == DIM && w2.size(2) == INTER / 2, "w2 [256, 4096, 256]");
    auto out = torch::empty({256}, torch::dtype(torch::kUInt8));
    CUtensorMap m[2];
    encode_weight_map(&m[0], w13.data_ptr(), static_cast<long long>(NEXP) * 2 * INTER, DIM / 2, NCH1);
    encode_weight_map(&m[1], w2.data_ptr(), static_cast<long long>(NEXP) * DIM, INTER / 2, NCH2);
    std::memcpy(out.data_ptr(), m, 256);
    return out;
}
// the shared expert's production FP8 weights (DeepGEMM, in place): gate_up [1024][4096], down [4096][512] (uint8 views)
torch::Tensor encode_maps_shared(torch::Tensor gu, torch::Tensor dn) {
    TORCH_CHECK(gu.is_cuda() && dn.is_cuda() && gu.element_size() == 1 && dn.element_size() == 1 && gu.is_contiguous() && dn.is_contiguous());
    TORCH_CHECK(gu.dim() == 2 && gu.size(0) == 2 * INTER && gu.size(1) == DIM && dn.dim() == 2 && dn.size(0) == DIM && dn.size(1) == INTER);
    auto out = torch::empty({256}, torch::dtype(torch::kUInt8));
    CUtensorMap m[2];
    {
        const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(2 * INTER), static_cast<cuuint64_t>(NCH1)};
        const cuuint64_t strides[2] = {static_cast<cuuint64_t>(DIM), 128};
        const cuuint32_t box[3] = {128, 128, 1}, es[3] = {1, 1, 1};
        TORCH_CHECK(encode_fn()(&m[0], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, gu.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    }
    {
        const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(DIM), static_cast<cuuint64_t>(NCH2)};
        const cuuint64_t strides[2] = {static_cast<cuuint64_t>(INTER), 128};
        const cuuint32_t box[3] = {128, 128, KC}, es[3] = {1, 1, 1};
        TORCH_CHECK(encode_fn()(&m[1], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, dn.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    }
    std::memcpy(out.data_ptr(), m, 256);
    return out;
}
static int g_pdl = 1;
void set_pdl(int64_t v) { g_pdl = static_cast<int>(v); }
// ws: dict-free flat argument list (see tpb2.py RankState): workspace tensors + LL peer pointers
void run(torch::Tensor maps, torch::Tensor w13_sf, torch::Tensor w2_sf, torch::Tensor lanes, torch::Tensor xb, c10::optional<torch::Tensor> glog,
         c10::optional<torch::Tensor> gbias, torch::Tensor ids, torch::Tensor rw, c10::optional<torch::Tensor> shared, double factor,
         torch::Tensor xq, torch::Tensor xs, torch::Tensor xg, torch::Tensor xgsf, c10::optional<torch::Tensor> dbg, torch::Tensor himg, torch::Tensor hsf, torch::Tensor part, torch::Tensor part_flags,
         torch::Tensor xflag, torch::Tensor gcnt, torch::Tensor pan_cnt, torch::Tensor rdy, torch::Tensor fc2_next, torch::Tensor rt_cnt, torch::Tensor work,
         torch::Tensor outp, std::vector<int64_t> rs, std::vector<int64_t> ag, torch::Tensor out, int64_t my_rank, int64_t ndev, int64_t grid,
         c10::optional<torch::Tensor> smaps, c10::optional<torch::Tensor> sfa1s, c10::optional<torch::Tensor> sfa2s, torch::Tensor xgs,
         torch::Tensor xgsfs, torch::Tensor hsimg, torch::Tensor hssf, torch::Tensor shout, torch::Tensor xsf2, torch::Tensor tcnt, std::vector<int64_t> wptr) {
    const int M = static_cast<int>(xb.size(0));
    TORCH_CHECK((!TPB2_MMAX || M <= TPB2_MMAX) && M >= TPB2_MMIN, "this build: TPB2_MMIN <= M <= TPB2_MMAX");
    TORCH_CHECK(M >= 1 && M <= MAXM && xb.size(1) == DIM && xb.scalar_type() == torch::kBFloat16 && xb.is_contiguous());
    TORCH_CHECK(ids.size(0) == M && ids.size(1) <= 6 && ids.scalar_type() == torch::kInt32 && rw.scalar_type() == torch::kFloat32);
    TORCH_CHECK(ndev == 1 || (ndev == NDEV && (rs.size() == NDEV || rs.size() == NDEV + 1) && (ag.size() == NDEV || ag.size() == NDEV + 1)));
    Params P;
    std::memset(&P, 0, sizeof(P));
    std::memcpy(&P.map1, maps.data_ptr(), 256);
    P.w13_sf = w13_sf.data_ptr<uint8_t>(); P.w2_sf = w2_sf.data_ptr<uint8_t>(); P.lanes = lanes.data_ptr<uint8_t>();
    P.xb = reinterpret_cast<const __nv_bfloat16*>(xb.data_ptr());
    P.glog = glog.has_value() ? glog->data_ptr<float>() : nullptr;
    P.gbias = gbias.has_value() ? gbias->data_ptr<float>() : nullptr;
    TORCH_CHECK(!P.glog || (P.gbias && ids.size(1) == 6), "fused gate: top-6 + bias");
    P.ids = ids.data_ptr<int>(); P.rw = rw.data_ptr<float>(); P.topk = static_cast<int>(ids.size(1));
    P.shared = shared.has_value() ? reinterpret_cast<const __nv_bfloat16*>(shared->data_ptr()) : nullptr;
    P.factor = static_cast<float>(factor);
    P.xq = xq.data_ptr<uint8_t>(); P.xs = xs.data_ptr<uint8_t>(); P.xg = xg.data_ptr<uint8_t>(); P.xgsf = xgsf.data_ptr<uint8_t>();
    P.dbg_ = dbg.has_value() ? reinterpret_cast<unsigned long long*>(dbg->data_ptr<int64_t>()) : nullptr; P.himg = himg.data_ptr<uint8_t>(); P.hsf = hsf.data_ptr<uint8_t>();
    P.part = part.data_ptr<float>(); P.part_flags = reinterpret_cast<unsigned*>(part_flags.data_ptr<int>());
    P.xflag = reinterpret_cast<unsigned*>(xflag.data_ptr<int>()); P.gcnt = reinterpret_cast<unsigned*>(gcnt.data_ptr<int>());
    P.pan_cnt = reinterpret_cast<unsigned*>(pan_cnt.data_ptr<int>()); P.rdy = reinterpret_cast<unsigned*>(rdy.data_ptr<int>()); P.fc2_next = reinterpret_cast<unsigned*>(fc2_next.data_ptr<int>());
    P.rt_cnt = reinterpret_cast<unsigned*>(rt_cnt.data_ptr<int>()); P.work = reinterpret_cast<unsigned*>(work.data_ptr<int>());
    P.outp = reinterpret_cast<__nv_bfloat16*>(outp.data_ptr());
    for (int d = 0; d < NDEV; ++d) {
        P.rs[d] = ndev == 1 ? nullptr : reinterpret_cast<uint4*>(rs[d]);
        P.ag[d] = ndev == 1 ? nullptr : reinterpret_cast<uint4*>(ag[d]);
    }
    P.rsmc = ndev > 1 && rs.size() > NDEV ? reinterpret_cast<uint4*>(rs[NDEV]) : nullptr;   // (rs / ag lists: NDEV peers [+ multicast])
    P.agmc = ndev > 1 && ag.size() > NDEV ? reinterpret_cast<uint4*>(ag[NDEV]) : nullptr;
    TORCH_CHECK(!(TPB2_MC & 5) || ndev == 1 || P.rsmc, "TPB2_MC 1 / 4: the rs list needs the multicast address (NDEV + 1 entries)");
    TORCH_CHECK(!(TPB2_MC & 2) || ndev == 1 || P.agmc, "TPB2_MC 2: the ag list needs the multicast address (NDEV + 1 entries)");
    P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    P.M = M; P.my_rank = static_cast<int>(my_rank); P.ndev = static_cast<int>(ndev);
    P.oneshot = M <= TPB2_ONESHOT_MAXM ? 1 : 0;
    P.shw = smaps.has_value() ? 1 : 0;
    if (P.shw) {
        TORCH_CHECK(sfa1s.has_value() && sfa2s.has_value() && !shared.has_value(), "fused shared expert: maps + SFA images, no external shared output");
        std::memcpy(&P.map1s, smaps->data_ptr(), 256);
        P.sfa1s = sfa1s->data_ptr<uint8_t>(); P.sfa2s = sfa2s->data_ptr<uint8_t>();
    }
    P.xgs = xgs.data_ptr<uint8_t>(); P.xgsfs = xgsfs.data_ptr<uint8_t>(); P.hsimg = hsimg.data_ptr<uint8_t>(); P.hssf = hssf.data_ptr<uint8_t>();
    P.shout = reinterpret_cast<__nv_bfloat16*>(shout.data_ptr());
    P.xsf2 = xsf2.data_ptr<uint8_t>(); P.tcnt = reinterpret_cast<unsigned*>(tcnt.data_ptr<int>());
    P.nparts = wptr.size() > 4 ? static_cast<int>(wptr[4]) : 1; P.pstride = static_cast<int>(xb.size(0)) * NEXP;   // (wptr[4]: router K-parts)
    for (int i = 0; i < 4; ++i) P.wp[i] = i < static_cast<int>(wptr.size()) ? reinterpret_cast<const uint8_t*>(wptr[i]) : nullptr;
    if (P.wp[0] && (!P.shw || !P.wp[2])) { P.wp[2] = P.wp[0]; P.wp[3] = P.wp[0]; }   // (no fused shared expert: harmless re-prefetch)
    if (wptr.size() == 6 && wptr[5]) {   // in-kernel dsv3 router (M <= 4): wptr[5] = router W bf16 [256][4096]; glog = rpart [M][256]
        TORCH_CHECK(M >= 1 && M <= 4 && P.glog && P.nparts == 1, "in-kernel dsv3 router: 1 <= M <= 4");
        P.rF = 1; P.rwf = reinterpret_cast<const __nv_bfloat16*>(wptr[5]); P.rpart = const_cast<float*>(P.glog);
    } else if (wptr.size() > 5 && wptr[5]) {   // in-kernel router: wptr[5] = router W bf16 [256][4096], wptr[6 ..] = (k0, kn) per K-part; glog = rpart
        const int S = P.nparts;
        TORCH_CHECK(S >= 1 && S <= 4 && static_cast<int>(wptr.size()) >= 6 + 2 * S && M >= 5 && M <= 128 && P.glog, "in-kernel router: 5 <= M <= 128");
        P.rS = S; P.rR = (M + 7) & ~7; P.rpart = const_cast<float*>(P.glog); P.rB = 1;
        if (TPB2_RRB && M > 64) { P.rB = (M + 63) / 64; P.rR = 64; }   // (64-row blocks: box height 64, OOB rows zero-filled)
        TORCH_CHECK(16 * S * P.rB <= static_cast<int>(grid), "in-kernel router: units > grid");
        int rch = 0;
        for (int i = 0; i < S; ++i) {
            P.rk0[i] = static_cast<int>(wptr[6 + 2 * i]); P.rkn[i] = static_cast<int>(wptr[7 + 2 * i]);
            TORCH_CHECK(P.rk0[i] % 64 == 0 && P.rkn[i] % 64 == 0 && P.rkn[i] <= 32 * 64);   // (W slice 16 x rkn bf16 <= 64 KB at RW_OFF)
            rch = std::max(rch, P.rkn[i] / 64);
        }
        P.rch = rch;
        P.rxch = P.rR * rch * 128 <= 96 * 1024 ? rch : (48 * 1024) / (P.rR * 128);   // (x sub-parts of <= 48 KB, double-buffered)
        const cuuint64_t sx[2] = {DIM * 2, 128};
        const cuuint32_t es[3] = {1, 1, 1};
        const cuuint64_t dx[3] = {64, static_cast<cuuint64_t>(M), DIM / 64}, dw[3] = {64, NEXP, DIM / 64};
        const cuuint32_t bx[3] = {64, static_cast<cuuint32_t>(P.rR), static_cast<cuuint32_t>(P.rxch)}, bw[3] = {64, 16, static_cast<cuuint32_t>(rch)};
        TORCH_CHECK(encode_fn()(&P.maprx, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, xb.data_ptr(), dx, sx, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
        TORCH_CHECK(encode_fn()(&P.maprw, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, reinterpret_cast<void*>(wptr[5]), dw, sx, bw, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    }
    TORCH_CHECK(tcnt.numel() >= static_cast<int64_t>(2 * MAXM * NRT + LL_INTS), "tcnt: counters + the T1LL images");
    {   // gather4 maps over the token-major x codes / SFB-layout scales (box height 1: four rows per gather4)
        const cuuint64_t d1[2] = {DIM, MAXM}, s1[1] = {DIM};
        const cuuint32_t b1[2] = {128, 1}, es[2] = {1, 1};
        TORCH_CHECK(encode_fn()(&P.mapxq, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, xq.data_ptr(), d1, s1, b1, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
        const cuuint64_t d2[2] = {NSTG1 * 16, MAXM}, s2[1] = {NSTG1 * 16};
        const cuuint32_t b2[2] = {16, 1};
        TORCH_CHECK(encode_fn()(&P.mapxsf, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, xsf2.data_ptr(), d2, s2, b2, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    }
    const int G = static_cast<int>(grid);
    TORCH_CHECK(G >= 1 && G <= MAXG);
    TORCH_CHECK(TPB2_SPEC != 1 || (P.ndev == NDEV && P.topk == 6 && P.shw && P.glog && !P.shared && (TPB2_SPEC_RF || !P.rF)), "TPB2_SPEC: TP4, top-6, fused gate + shared expert");
    TORCH_CHECK(TPB2_SPEC != 2 || (P.ndev == NDEV && P.topk == 6 && !P.shw && !P.glog && !P.shared && !P.rF && P.nparts == 1), "TPB2_SPEC=2: TP size NDEV, top-6, routed only (ids / weights given)");
    TORCH_CHECK(TPB2_RS != 0 || (P.rS == 0 && (!TPB2_SPEC || P.nparts == 1)), "TPB2_RS=0: no in-kernel router");
    TORCH_CHECK(!TPB2_GRID || G == TPB2_GRID, "TPB2_GRID: grid");
    TORCH_CHECK(TPB2_RANK < 0 || P.my_rank == TPB2_RANK, "TPB2_RANK: rank");
    {   // (RR_ALWAYS: every shared piece on its own unit)
        const int shq = (TPB2_SHW1 && M >= TPB2_SHW1) ? 1 : TPB2_SHQ;
        TORCH_CHECK(!P.shw || ((M + NMAX - 1) / NMAX) * (INTER / 128) * shq <= G, "shared pieces > grid");
    }
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tpb2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC)); attr = true; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = SMEM_ALLOC; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = (TPB2_PAIR || TPB2_CSPLIT) ? 2 : 1; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = (TPB2_PAIR || TPB2_CSPLIT) ? 2 : 1;
    TORCH_CHECK(!(TPB2_PAIR || TPB2_CSPLIT) || G % 2 == 0, "PAIR / CSPLIT: an even grid");
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tpb2_kernel, P));
}
int64_t smem_alloc() { return SMEM_ALLOC; }

// sglang dsv3_router_gemm (M <= 4) reproduced bit for bit -- 128 threads per expert, 8 bf16 per thread per K iteration of 1024, FMA chains,
// warp xor-sum 16 .. 1, then 0 + the 4 warp sums in order -- but with the W row loaded before the PDL wait and the dependents (the MoE
// kernel) triggered right after it: the MoE kernel's launch and pre-wait prologue overlap the router (it still waits for its completion)
__global__ void __launch_bounds__(128) dsv3_router_early(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int M) {
    const int e = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float red[4][4];
    uint4 wv[DIM / 1024];
    const uint4* wr = reinterpret_cast<const uint4*>(w + static_cast<size_t>(e) * DIM) + tid;
#pragma unroll
    for (int i = 0; i < DIM / 1024; ++i) wv[i] = __ldg(wr + 128 * i);
    griddep_wait();
    griddep_launch();
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4* xr = reinterpret_cast<const uint4*>(x) + tid;
#pragma unroll
    for (int i = 0; i < DIM / 1024; ++i) {
        uint4 xv[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) xv[m] = m < M ? __ldcg(xr + m * (DIM / 8) + 128 * i) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t w4[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t a[4] = {xv[m].x, xv[m].y, xv[m].z, xv[m].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[m] = __fmaf_rn(bf16lo(a[j]), bf16lo(w4[j]), acc[m]);
                acc[m] = __fmaf_rn(bf16hi(a[j]), bf16hi(w4[j]), acc[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) acc[m] = acc[m] + __shfl_xor_sync(0xffffffffu, acc[m], o);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < 4; ++m) red[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0 && lane < M) {
        float fs = 0.f;
#pragma unroll
        for (int w2 = 0; w2 < 4; ++w2) fs += red[lane][w2];
        out[lane * NEXP + e] = fs;
    }
}
void router_dsv3(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    const int M = static_cast<int>(x.size(0));
    TORCH_CHECK(M >= 1 && M <= 4 && x.size(1) == DIM && w.size(0) == NEXP && w.size(1) == DIM && out.numel() >= M * NEXP);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NEXP); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = 0; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = g_pdl;
    cfg.attrs = at; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsv3_router_early, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), out.data_ptr<float>(), M));
}
#endif  // B2_NO_TORCH

}  // namespace TPB2_NS

#ifndef B2_NO_TORCH
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_maps", &TPB2_NS::encode_maps);
    m.def("encode_maps_shared", &TPB2_NS::encode_maps_shared);
    m.def("run", &TPB2_NS::run);
    m.def("set_pdl", &TPB2_NS::set_pdl);
    m.def("router_dsv3", &TPB2_NS::router_dsv3);
    m.def("smem_alloc", &TPB2_NS::smem_alloc);
    m.attr("ABI") = "xsf2,tcnt,wptr";
    m.attr("LL_INTS") = static_cast<int64_t>(TPB2_NS::LL_INTS);
    m.attr("OUTP_LD") = TPB2_NS::OUTP_LD;
    m.attr("LLS") = static_cast<int64_t>(TPB2_LLS);
    m.attr("MAXM") = TPB2_NS::MAXM; m.attr("NDEV") = TPB2_NS::NDEV; m.attr("MAXP") = TPB2_NS::MAXP; m.attr("MAXG") = TPB2_NS::MAXG;
    m.attr("MAXPS") = TPB2_NS::MAXPS; m.attr("NMAX") = TPB2_NS::NMAX; m.attr("NRT") = TPB2_NS::NRT; m.attr("NMSG") = TPB2_NS::NMSG; m.attr("NCH2") = TPB2_NS::NCH2;
    m.attr("NSTG2") = TPB2_NS::NSTG2; m.attr("NCH1") = TPB2_NS::NCH1; m.attr("NSTG1") = TPB2_NS::NSTG1; m.attr("RS_PAR") = static_cast<int64_t>(TPB2_NS::RS_PAR); m.attr("AG_PAR") = static_cast<int64_t>(TPB2_NS::AG_PAR);
}
#endif
