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
// ANCHOR probes: thread 0 of the CTA
__device__ __forceinline__ void xt_anchor(void* buf, bool end) {
    if (!XT_ON || !buf) return;
    uint64_t* h = reinterpret_cast<uint64_t*>(buf) + static_cast<size_t>(blockIdx.x) * XT_HDR;
    const uint64_t g = xt_gtime(), c = xt_clock64();
    h[end ? 2 : 0] = g; h[end ? 3 : 1] = c;
    if (!end) h[4] = xt_smid();
}
