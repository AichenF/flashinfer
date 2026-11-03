// xtrace.cuh -- intra-kernel phase tracing after Xtrace (arXiv 2609.28769), source / PTX tier.
//
//   INIT probe   (kernel entry, every warp): the warp's ring-buffer slot address + its record counter, kept in registers (3 x 32 bit).
//   RECORD probe (one lane of the warp): the per-SM cycle counter (%clock, ~10 ns to read, one register) and the site id + two
//                 arguments, one 16-B store into the warp's ring (XT_DEPTH records; the oldest are overwritten).
//   ANCHOR probe (thread 0 at CTA start and end): %globaltimer and %clock64 read back to back, + %smid: the host maps every record's
//                 SM cycles onto the GPU-global nanosecond timeline (per-SM clocks are not synchronized across SMs).
//
// Buffer (device, zeroed before the launch): [grid][XT_HDR] uint64 CTA headers {gt0, clk0, gt1, clk1, smid, 0, 0, 0}, then
// [grid][XT_WARPS][XT_DEPTH] uint4 records {clock lo 32, site, a0, a1}.  Decoded / analyzed / rendered by xtrace.py.
// Site ids: bits 0..11 the site, bits 12..13 the kind (0 point, 1 scope begin, 2 scope end); see xtrace.py SITES.
#pragma once
#include <cstdint>

#ifndef XT_ON
#define XT_ON 0
#endif
// XT_FMT 2 (XTrace layout, after the gwatch / xtrace-profile warp schema): one 8-B record per event {SM clock (32 bit), site 7 bit |
// kind 2 bit | payload 23 bit}, written by one lane of the warp into its own BOUNDED slab (32-B header {magic, smid, cta, warp,
// overflow, ..} + XT_DEPTH records).  No ring overwrite: the first record past XT_DEPTH sets the overflow flag and the trace is
// rejected by the validator.  XT_MASK selects the site groups compiled in (small causal groups = small perturbation per run).
#ifndef XT_FMT
#define XT_FMT 1
#endif
#ifndef XT_MASK
#define XT_MASK 0xFFu
#endif
#define XT_SLAB_HDR 4   // uint64 words of a v2 warp header
// site groups (K-stage roles): 0 prologue, 1 producer (empty wait / TMA issue), 2 MMA (full wait / issue..commit / acc wait),
// 3 B operand, 4 epilogue, 5 tail (T1 / all-reduce)
__host__ __device__ constexpr uint32_t xt_group(uint32_t site) {
    return ((site & 0xFFFu) >= 10u && (site & 0xFFFu) <= 14u) ? 1u
         : (((site & 0xFFFu) >= 20u && (site & 0xFFFu) <= 24u) || (site & 0xFFFu) == 71u || (site & 0xFFFu) == 72u) ? 2u
         : ((site & 0xFFFu) == 30u || (site & 0xFFFu) == 31u) ? 3u
         : (((site & 0xFFFu) >= 40u && (site & 0xFFFu) <= 49u) || ((site & 0xFFFu) >= 73u && (site & 0xFFFu) <= 75u)) ? 4u
         : ((site & 0xFFFu) >= 50u && (site & 0xFFFu) <= 59u) ? 5u : 0u;
}
#ifndef XT_DEPTH
#define XT_DEPTH 1024
#endif
#ifndef XT_WARPS
#define XT_WARPS 12
#endif
#define XT_HDR 8
#define XT_PT(s) (s)
#define XT_BEG(s) ((s) | 0x1000u)
#define XT_END(s) ((s) | 0x2000u)

__device__ __forceinline__ uint32_t xt_clock() { uint32_t c; asm volatile("mov.u32 %0, %%clock;" : "=r"(c)); return c; }
__device__ __forceinline__ uint64_t xt_gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t xt_clock64() { uint64_t c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)); return c; }
__device__ __forceinline__ uint32_t xt_smid() { uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

#if XT_FMT == 2
struct Xt {
    uint2* p;      // this warp's records (null: tracing off)
    uint32_t* h;   // its header
    uint32_t n;    // records attempted by this lane
    // INIT probe: every warp, once (lane 0 writes the header identity)
    __device__ __forceinline__ void init(void* buf) {
        p = nullptr; h = nullptr; n = 0u;
        if (!XT_ON || !buf) return;
        const uint32_t w = threadIdx.x >> 5;
        if (w >= XT_WARPS) return;
        uint64_t* slab = reinterpret_cast<uint64_t*>(buf) + static_cast<size_t>(gridDim.x) * XT_HDR
                       + (static_cast<size_t>(blockIdx.x) * XT_WARPS + w) * (XT_SLAB_HDR + XT_DEPTH);
        h = reinterpret_cast<uint32_t*>(slab);
        p = reinterpret_cast<uint2*>(slab + XT_SLAB_HDR);
        if ((threadIdx.x & 31u) == 0u) { h[0] = 0x58543202u; h[1] = xt_smid(); h[2] = blockIdx.x; h[3] = w; }
    }
    // RECORD probe: called by one lane; sites outside XT_MASK compile to nothing (the site is a constant at every call site)
    __device__ __forceinline__ void rec(uint32_t site, uint32_t a0 = 0u, uint32_t a1 = 0u) {
        if (!XT_ON || !((XT_MASK >> xt_group(site)) & 1u) || !p) return;
        (void)a1;
        if (n < XT_DEPTH) {
            const uint32_t meta = (site & 0x7Fu) | (((site >> 12) & 3u) << 7) | ((((a0 >> 28) & 7u) << 20 | (a0 & 0xFFFFFu)) << 9);
            p[n] = make_uint2(xt_clock(), meta);
        } else if (n == XT_DEPTH) {
            h[4] = 1u;   // overflow: this warp's history is incomplete -> the trace is rejected
        }
        ++n;
    }
};
#else
struct Xt {
    uint4* p;     // this warp's ring (null: tracing off)
    uint32_t n;   // records written by this lane
    // INIT probe: every warp, once
    __device__ __forceinline__ void init(void* buf) {
        p = nullptr; n = 0u;
        if (!XT_ON || !buf) return;
        const uint32_t w = threadIdx.x >> 5;
        if (w >= XT_WARPS) return;
        uint4* recs = reinterpret_cast<uint4*>(reinterpret_cast<uint64_t*>(buf) + static_cast<size_t>(gridDim.x) * XT_HDR);
        p = recs + (static_cast<size_t>(blockIdx.x) * XT_WARPS + w) * XT_DEPTH;
    }
    // RECORD probe: called by one lane
    __device__ __forceinline__ void rec(uint32_t site, uint32_t a0 = 0u, uint32_t a1 = 0u) {
        if (!XT_ON || !p) return;
        p[n & (XT_DEPTH - 1)] = make_uint4(xt_clock(), site, a0, a1);
        ++n;
    }
};
#endif
// ANCHOR probes: thread 0 of the CTA
__device__ __forceinline__ void xt_anchor(void* buf, bool end) {
    if (!XT_ON || !buf) return;
    uint64_t* h = reinterpret_cast<uint64_t*>(buf) + static_cast<size_t>(blockIdx.x) * XT_HDR;
    const uint64_t g = xt_gtime(), c = xt_clock64();
    h[end ? 2 : 0] = g; h[end ? 3 : 1] = c;
    if (!end) h[4] = xt_smid();
}
