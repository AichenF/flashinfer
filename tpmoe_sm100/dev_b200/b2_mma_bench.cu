// B200 microbenchmark: tcgen05.mma kind::mxf8f6f4.block_scale.block32 (M128, e2m1 A x e4m3 B, K32 per instruction) issue / execution
// rate from one CTA per SM, operands static in shared memory (no loads): ns per MMA by N, MMAs per commit (stage), with / without the
// per-stage SF copies (tcgen05.cp 32x128b.warpx4) and with cta_group::1.  Mirrors the tpmoe_b2 MMA warp (elected lane, converged warp).
#ifndef B2_NO_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace b2mb {

__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
    return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
           (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (static_cast<uint64_t>(layout) << 61);
}
__device__ __forceinline__ uint32_t idesc_mx(int n, uint32_t sf_id, uint32_t atype) {
    // (atype bits 0-7: A type, bits 8-15: B type (default e4m3 = 0), bits 16+: MMA M (default 128))
    const uint32_t m = (atype >> 16) ? (atype >> 16) : 128u;
    return (sf_id << 4) | ((atype & 0xFFu) << 7) | (((atype >> 8) & 0xFFu) << 10) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) |
           ((m >> 4) << 24) | (sf_id << 29);
}
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
                 " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
                 ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, uint64_t desc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(desc) : "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ bool elect_one() {
    uint32_t pred = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
    return pred != 0;
}

// smem: A 4 x 16 KB chunks (64 KB), B 4 x 8 KB (N <= 64 rows x 128 B), SF images 8 x 512 B; contents: zeros (values do not matter)
constexpr int SM_A = 0, SM_B = 65536, SM_SF = SM_B + 4 * 8192, SM_BAR = SM_SF + 8 * 512, SMEM = SM_BAR + 64 + 1024;

// per round: `per` MMAs (K32 each over the 4 A chunks round-robin) into one accumulator, then (cp) 3 SF copies, commit, wait the commit
// (wait = 1: serialized, measures issue + execution latency per round; wait = 0: the commit of round r is waited at round r + depth)
__global__ void __launch_bounds__(128, 1) mma_bench(int n, int per, int rounds, int cp, int depth, int atype, int ilv, unsigned long long* out) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    uint8_t* base = smem_raw + (sb - sraw);
    __shared__ uint32_t tslot;
    const int tid = threadIdx.x, warp = tid >> 5;
    if (atype & 0x80000000) {   // random operand bytes (unpacked e2m1 keeps bits 0-3 / e4m3 avoids NaN / inf: 0x77 mask), SF 1.0 (0x7F)
        for (int i = tid; i < SM_BAR / 4; i += 128) {
            uint32_t h = static_cast<uint32_t>(i) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
            reinterpret_cast<uint32_t*>(base)[i] = i * 4 >= SM_SF ? 0x7F7F7F7Fu : (h & ((atype & 0x40000000) ? 0x0F0F0F0Fu : 0x77777777u));
        }
    } else
    for (int i = tid; i < (SM_BAR) / 16; i += 128) reinterpret_cast<uint4*>(base)[i] = make_uint4(0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu);
    atype &= 0x3FFFFFFF;
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) mbar_init(sb + SM_BAR + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tslot;
    if (warp == 1) {
        const uint32_t idb = idesc_mx(n, 0, static_cast<uint32_t>(atype));
        const uint64_t ad = smem_desc(sb + SM_A, 0, 1024, 2), bd = smem_desc(sb + SM_B, 0, 1024, 2);
        unsigned long long t0 = 0;
        for (int r = 0; r < rounds + 8; ++r) {
            if (r == 8) t0 = clock64();   // (8 warm-up rounds)
            const int slot = r % 8;
            if (r >= depth) mbar_wait(sb + SM_BAR + 8 * ((r - depth) % 8), ((r - depth) / 8) & 1);
            tc_fence_after();
            if (elect_one()) {
                if (cp && n > 64) {
                    tc_cp_sf(tmem + 256, smem_desc(sb + SM_SF, 0, 128, 0));
                    tc_cp_sf(tmem + 384, smem_desc(sb + SM_SF + 512, 0, 128, 0));
                    tc_cp_sf(tmem + 388, smem_desc(sb + SM_SF + 1024, 0, 128, 0));
                } else if (cp) {
                    tc_cp_sf(tmem + 256, smem_desc(sb + SM_SF, 0, 128, 0));
                    tc_cp_sf(tmem + 260, smem_desc(sb + SM_SF + 512, 0, 128, 0));
                    tc_cp_sf(tmem + 272, smem_desc(sb + SM_SF + 1024, 0, 128, 0));
                }
                // ilv independent accumulators (MMA j -> accumulator j % ilv: ilv chains interleaved), the set alternating per round
                const int colw = n <= 32 ? 32 : 64;
                for (int j = 0; j < per; ++j) {
                    const int c = (j >> 2) & 3, kb = j & 3;
                    if (n > 64)   // (wide N: one accumulator at columns 0 .. n - 1, SFA 256, SFB 384; per <= 4 so B stays in its 32 KB)
                        tc_mma(tmem, ad + 2 * kb, bd + 2 * kb, idb | (kb << 4) | (kb << 29), tmem + 256, tmem + 384, j > 0);
                    else
                    tc_mma(tmem + ((j % ilv) + (r & 1) * ilv) * colw, ad + c * (16384 >> 4) + 2 * kb, bd + c * ((n * 128) >> 4) + 2 * kb,
                           idb | (kb << 4) | (kb << 29), tmem + 256 + 4 * (c & 1), tmem + 272, j >= ilv);
                }
                tc_commit(sb + SM_BAR + 8 * slot);
            }
            __syncwarp();
        }
        for (int r = rounds + 8 - depth; r < rounds + 8; ++r) if (r >= 0) mbar_wait(sb + SM_BAR + 8 * (r % 8), (r / 8) & 1);
        const unsigned long long t1 = clock64();
        if (tid == 32) out[blockIdx.x] = t1 - t0;
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 0) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(512) : "memory");
    }
}

// cta_group::2 variant: CTA pair (cluster of 2), the leader issues M = 256 MMAs (each CTA's own 128 A rows, N / 2 B rows each)
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) mma_bench2(int n, int per, int rounds, int cp, int depth, unsigned long long* out) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    uint8_t* base = smem_raw + (sb - sraw);
    __shared__ uint32_t tslot;
    const int tid = threadIdx.x, warp = tid >> 5;
    const bool leader = cluster_rank() == 0;
    for (int i = tid; i < (SM_BAR) / 16; i += 128) reinterpret_cast<uint4*>(base)[i] = make_uint4(0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu);
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) mbar_init(sb + SM_BAR + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    tc_fence_before();
    cluster_sync();
    tc_fence_after();
    const uint32_t tmem = tslot;
    if (warp == 1 && leader) {
        const uint32_t idb = (5u << 7) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) | ((256u >> 4) << 24);
        const uint64_t ad = smem_desc(sb + SM_A, 0, 1024, 2), bd = smem_desc(sb + SM_B, 0, 1024, 2);
        unsigned long long t0 = 0;
        for (int r = 0; r < rounds + 8; ++r) {
            if (r == 8) t0 = clock64();
            const int slot = r % 8;
            if (r >= depth) mbar_wait(sb + SM_BAR + 8 * ((r - depth) % 8), ((r - depth) / 8) & 1);
            tc_fence_after();
            if (elect_one()) {
                if (cp) {
                    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(tmem + 128), "l"(smem_desc(sb + SM_SF, 0, 128, 0)) : "memory");
                    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(tmem + 132), "l"(smem_desc(sb + SM_SF + 512, 0, 128, 0)) : "memory");
                    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(tmem + 144), "l"(smem_desc(sb + SM_SF + 1024, 0, 128, 0)) : "memory");
                }
                for (int j = 0; j < per; ++j) {
                    const int c = (j >> 2) & 3, kb = j & 3;
                    const uint32_t idk = idb | (kb << 4) | (kb << 29), acc = j != 0;
                    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
                                 " tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
                                 ::"r"(tmem + (r & 1) * 64), "l"(ad + c * (16384 >> 4) + 2 * kb), "l"(bd + c * (((n / 2) * 128) >> 4) + 2 * kb), "r"(idk),
                                   "r"(tmem + 128 + 4 * (c & 1)), "r"(tmem + 144), "r"(acc) : "memory");
                }
                asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                             ::"r"(sb + SM_BAR + 8 * slot), "h"(static_cast<uint16_t>(3)) : "memory");
            }
            __syncwarp();
        }
        for (int r = rounds + 8 - depth; r < rounds + 8; ++r) if (r >= 0) mbar_wait(sb + SM_BAR + 8 * (r % 8), (r / 8) & 1);
        const unsigned long long t1 = clock64();
        if (tid == 32) out[blockIdx.x] = t1 - t0;
    }
    tc_fence_before();
    cluster_sync();
    if (warp == 0) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256) : "memory");
    }
}

// Variant probe: what costs ~250 ns per round?  mode bits: 1 = no tcgen05.fence::after_thread_sync per round, 2 = one elect for the whole
// run (the elected lane loops over all rounds; no per-round elect / __syncwarp), 4 = no mbarrier wait at all (only at the end);
// commits every `cper` rounds (the others issue MMAs only).  PAIR: cta_group::2, M256 (leader issues).
template <bool PAIR>
__global__ void __launch_bounds__(128, 1) mma_bench3(int n, int per, int rounds, int cper, int mode, unsigned long long* out) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    uint8_t* base = smem_raw + (sb - sraw);
    __shared__ uint32_t tslot;
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t crank = 0;
    if (PAIR) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    for (int i = tid; i < (SM_BAR) / 16; i += 128) reinterpret_cast<uint4*>(base)[i] = make_uint4(0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu);
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) mbar_init(sb + SM_BAR + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        if (PAIR) {
            asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(256) : "memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
        } else {
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(256) : "memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        }
    }
    tc_fence_before();
    if (PAIR) asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    else __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tslot;
    if (warp == 1 && crank == 0) {
        const uint32_t idb = (5u << 7) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) | ((PAIR ? 256u : 128u) >> 4 << 24);
        const uint64_t ad = smem_desc(sb + SM_A, 0, 1024, 2), bd = smem_desc(sb + SM_B, 0, 1024, 2);
        const int brows = PAIR ? n / 2 : n;
        unsigned long long t0 = 0;
        int ncommit = 0;
        auto commit_to = [&](uint32_t bb) {
            if (PAIR)
                asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                             ::"r"(bb), "h"(static_cast<uint16_t>(3)) : "memory");
            else
                tc_commit(bb);
        };
        auto round = [&](int r) {
            for (int j = 0; j < per; ++j) {
                const int c = (j >> 2) & 3, kb = j & 3;
                const uint32_t idk = idb | (kb << 4) | (kb << 29), acc = j != 0;
                const uint32_t d = tmem + (r & 1) * 64;
                if (PAIR)
                    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
                                 " tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
                                 ::"r"(d), "l"(ad + c * (16384 >> 4) + 2 * kb), "l"(bd + c * ((brows * 128) >> 4) + 2 * kb), "r"(idk),
                                   "r"(tmem + 128 + 4 * (c & 1)), "r"(tmem + 144), "r"(acc) : "memory");
                else
                    tc_mma(d, ad + c * (16384 >> 4) + 2 * kb, bd + c * ((brows * 128) >> 4) + 2 * kb, idk, tmem + 128 + 4 * (c & 1), tmem + 144, acc);
            }
            if ((r + 1) % cper == 0) {
                commit_to(sb + SM_BAR + 8 * (ncommit % 7));
                ++ncommit;
            }
        };
        if (mode & 2) {
            if (elect_one()) {
                for (int r = 0; r < rounds + 8; ++r) {
                    if (r == 8) t0 = clock64();
                    if (!(mode & 4) && ncommit >= 4 && (r % cper) == 0) mbar_wait(sb + SM_BAR + 8 * ((ncommit - 4) % 7), ((ncommit - 4) / 7) & 1);
                    if (!(mode & 1)) tc_fence_after();
                    round(r);
                }
                commit_to(sb + SM_BAR + 8 * 7);
                mbar_wait(sb + SM_BAR + 8 * 7, 0);
                out[blockIdx.x] = clock64() - t0;
            }
            __syncwarp();
        } else {
            for (int r = 0; r < rounds + 8; ++r) {
                if (r == 8) t0 = clock64();
                if (!(mode & 4) && ncommit >= 4 && (r % cper) == 0) mbar_wait(sb + SM_BAR + 8 * ((ncommit - 4) % 7), ((ncommit - 4) / 7) & 1);
                if (!(mode & 1)) tc_fence_after();
                if (elect_one()) round(r);
                ncommit = __shfl_sync(0xffffffffu, ncommit, __ffs(__activemask()) - 1);   // (all lanes track it)
                __syncwarp();
            }
            if (elect_one()) commit_to(sb + SM_BAR + 8 * 7);
            __syncwarp();
            mbar_wait(sb + SM_BAR + 8 * 7, 0);
            if (tid == 32) out[blockIdx.x] = clock64() - t0;
        }
    }
    tc_fence_before();
    if (PAIR) asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    else __syncthreads();
    if (warp == 0) {
        tc_fence_after();
        if (PAIR) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256) : "memory");
        else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256) : "memory");
    }
}

// Clean issue (the tpmoe_b2 MMA warp's code shape): per round, wait the commit of round r - depth (depth 0: never), fence, then PER MMAs
// fully unrolled with warp-uniform operands (descriptors computed before the elect), commit.  Stage buffers rotate over NSB smem stages
// (A 32 KB = 2 x 16 KB chunks, B N x 128 x 2) like the kernel's ring; accumulator per round alternates.  PAIR: cta_group::2 (M256).
template <int PER, bool PAIR>
__global__ void __launch_bounds__(128, 1) mma_bench4(int n, int rounds, int depth, unsigned long long* out) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    uint8_t* base = smem_raw + (sb - sraw);
    __shared__ uint32_t tslot;
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t crank = 0;
    if (PAIR) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    for (int i = tid; i < (SM_BAR) / 16; i += 128) reinterpret_cast<uint4*>(base)[i] = make_uint4(0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x3C3C3C3Cu);
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) mbar_init(sb + SM_BAR + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        if (PAIR) {
            asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(256) : "memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
        } else {
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(256) : "memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        }
    }
    tc_fence_before();
    if (PAIR) asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    else __syncthreads();
    tc_fence_after();
    const uint32_t tmem = __shfl_sync(0xffffffffu, tslot, 0);
    if (warp == 1 && crank == 0) {
        const int lane = tid & 31;
        const uint32_t nn = __shfl_sync(0xffffffffu, static_cast<uint32_t>(n), 0);
        const uint32_t idb = (5u << 7) | ((nn >> 3) << 17) | (1u << 23) | ((PAIR ? 256u : 128u) >> 4 << 24);
        const uint32_t brows = PAIR ? nn / 2 : nn;
        unsigned long long t0 = 0;
        for (int r = 0; r < rounds + 8; ++r) {
            if (r == 8) t0 = clock64();
            if (depth > 0 && r >= depth) mbar_wait(sb + SM_BAR + 8 * ((r - depth) % 8), ((r - depth) / 8) & 1);
            tc_fence_after();
            const uint32_t st = static_cast<uint32_t>(r) & 1u;   // 2 stage buffers of 32 KB A (within the 64 KB A region)
            const uint64_t ad0 = smem_desc(sb + SM_A + st * 32768, 0, 1024, 2), bd0 = smem_desc(sb + SM_B + st * 16384, 0, 1024, 2);
            const uint32_t dacc = tmem + (static_cast<uint32_t>(r) & 1u) * 64, tsf = tmem + 128;
            if (elect_one()) {
#pragma unroll
                for (int j = 0; j < PER; ++j) {
                    const int c = (j >> 2) & 1, kb = j & 3;
                    const uint32_t idk = idb | (kb << 4) | (kb << 29);
                    const uint64_t ac = ad0 + c * (16384 >> 4) + 2 * kb, bc = bd0 + c * ((brows * 128) >> 4) + 2 * kb;
                    if (PAIR)
                        asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
                                     " tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
                                     ::"r"(dacc), "l"(ac), "l"(bc), "r"(idk), "r"(tsf + 4 * c), "r"(tsf + 16), "r"(static_cast<uint32_t>(j != 0)) : "memory");
                    else
                        tc_mma(dacc, ac, bc, idk, tsf + 4 * c, tsf + 16, j != 0);
                }
                if (PAIR)
                    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                                 ::"r"(sb + SM_BAR + 8 * (r % 8)), "h"(static_cast<uint16_t>(3)) : "memory");
                else
                    tc_commit(sb + SM_BAR + 8 * (r % 8));
            }
            __syncwarp();
        }
        const int last = rounds + 7;
        mbar_wait(sb + SM_BAR + 8 * (last % 8), (last / 8) & 1);
        if (lane == 0) out[blockIdx.x] = clock64() - t0;
    }
    tc_fence_before();
    if (PAIR) asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    else __syncthreads();
    if (warp == 0) {
        tc_fence_after();
        if (PAIR) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256) : "memory");
        else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256) : "memory");
    }
}

// bench5 = bench4 (1-CTA, PER = 8) + (cp) 3 tcgen05.cp per round like the kernel + (tma) warp 2 streams 16U4 weight tiles by TMA into its
// own 4-stage ring (128 KB) while warp 1 issues MMAs from a fixed 32 KB A stage
struct B5 { CUtensorMap map; int tiles; };
__global__ void __launch_bounds__(128, 1) mma_bench5(const __grid_constant__ B5 A5, int n, int rounds, int cp, int tma, unsigned long long* out, int accm) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    __shared__ uint32_t tslot;
    __shared__ __align__(8) uint64_t tb_full[4], tb_empty[4];
    __shared__ volatile int stop, pfills, pdone;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t ring = sb + 65536;   // 4 x 32 KB TMA ring after the MMA operands (A 64 KB region reused: A at sb, B at sb + 32 KB)
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) mbar_init(sb + 65536 + 131072 + 8 * i, 1);
        for (int i = 0; i < 4; ++i) { mbar_init(su32(&tb_full[i]), 1); mbar_init(su32(&tb_empty[i]), 1); }
        stop = 0; pfills = 0; pdone = 0;
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = __shfl_sync(0xffffffffu, tslot, 0);
    const uint32_t bars = sb + 65536 + 131072;
    if (warp == 1) {
        const uint32_t nn = __shfl_sync(0xffffffffu, static_cast<uint32_t>(n), 0);
        const uint32_t idb = (5u << 7) | ((nn >> 3) << 17) | (1u << 23) | ((128u >> 4) << 24);
        unsigned long long t0 = 0;
        for (int r = 0; r < rounds + 8; ++r) {
            if (r == 8) t0 = clock64();
            if (r >= 4) mbar_wait(bars + 8 * ((r - 4) % 8), ((r - 4) / 8) & 1);
            tc_fence_after();
            const uint64_t ad0 = smem_desc(sb, 0, 1024, 2), bd0 = smem_desc(sb + 32768, 0, 1024, 2);
            // accm 0: accumulator alternates per round, reset each round; 1: one accumulator, chained over 16 rounds (an FC1 item); 2: two
            // items interleaved (chains on accumulators r & 1, each chained over its 16 rounds)
            const uint32_t dacc = tmem + (accm == 1 ? 0u : (static_cast<uint32_t>(r) & 1u) * 64), tsf = tmem + 128 + (static_cast<uint32_t>(r) & 3u) * 32;
            const bool reset = accm == 0 ? true : (accm == 1 ? (r % 16) == 0 : ((r >> 1) % 16) == 0);
            if (elect_one()) {
                if (cp) {
                    tc_cp_sf(tsf, smem_desc(sb + 49152, 0, 128, 0));
                    tc_cp_sf(tsf + 4, smem_desc(sb + 49152 + 512, 0, 128, 0));
                    tc_cp_sf(tsf + 16, smem_desc(sb + 49152 + 1024, 0, 128, 0));
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int c = (j >> 2) & 1, kb = j & 3;
                    tc_mma(dacc, ad0 + c * (16384 >> 4) + 2 * kb, bd0 + c * ((nn * 128) >> 4) + 2 * kb, idb | (kb << 4) | (kb << 29), tsf + 4 * c, tsf + 16,
                           !(reset && j == 0));
                }
                tc_commit(bars + 8 * (r % 8));
            }
            __syncwarp();
        }
        const int last = rounds + 7;
        mbar_wait(bars + 8 * (last % 8), (last / 8) & 1);
        if (lane == 0) { out[blockIdx.x] = clock64() - t0; stop = 1; }
    } else if (warp == 2 && tma && lane == 0) {   // TMA stream until the MMA warp is done (stop), every issued stage freed by warp 3
        int fills = 0;
        for (int t = blockIdx.x; !stop; t += gridDim.x) {
            if (t >= A5.tiles) t -= A5.tiles;
            for (int s = 0; s < 16 && !stop; ++s) {
                const int st = fills % 4;
                if (fills >= 4) mbar_wait(su32(&tb_empty[st]), ((fills / 4) - 1) & 1);
                asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(su32(&tb_full[st])), "r"(16384) : "memory");
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                             ::"r"(ring + st * 32768), "l"(&A5.map), "r"(0), "r"(t * 128), "r"(2 * s), "r"(su32(&tb_full[st])) : "memory");
                ++fills;
                pfills = fills;
            }
        }
        pdone = 1;
    } else if (warp == 3 && tma && lane == 0) {
        int c = 0;
        for (;;) {
            if (c < pfills) {
                mbar_wait(su32(&tb_full[c % 4]), (c / 4) & 1);
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(&tb_empty[c % 4])) : "memory");
                ++c;
            } else if (pdone) {
                if (c >= pfills) break;
            }
        }
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 0) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256) : "memory");
    }
}

#ifndef B2_NO_TORCH
torch::Tensor run2(int64_t n, int64_t per, int64_t rounds, int64_t cp, int64_t depth, int64_t grid) {
    auto out = torch::zeros({grid}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaFuncSetAttribute(mma_bench2, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    mma_bench2<<<static_cast<int>(grid), 128, SMEM, at::cuda::getCurrentCUDAStream()>>>(static_cast<int>(n), static_cast<int>(per), static_cast<int>(rounds),
                                                                                      static_cast<int>(cp), static_cast<int>(depth),
                                                                                      reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    return out;
}
template <int PER, bool PAIR>
static void launch4(int n, int rounds, int depth, int grid, unsigned long long* o) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = SMEM; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = PAIR ? 2 : 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaFuncSetAttribute(mma_bench4<PER, PAIR>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaLaunchKernelEx(&cfg, mma_bench4<PER, PAIR>, n, rounds, depth, o);
}
using EncodeFn5 = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                               const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
torch::Tensor run5(torch::Tensor w, int64_t n, int64_t rounds, int64_t cp, int64_t tma, int64_t grid, int64_t accm) {
    void* p = nullptr; cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    B5 A5; std::memset(&A5, 0, sizeof(A5));
    const long long rows = w.size(0);
    const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(rows), 32}, strides[2] = {2048, 64};
    const cuuint32_t box[3] = {128, 128, 2}, es[3] = {1, 1, 1};
    TORCH_CHECK(reinterpret_cast<EncodeFn5>(p)(&A5.map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, w.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
    A5.tiles = static_cast<int>(rows / 128);
    const int smem = 65536 + 131072 + 64 + 1024;
    cudaFuncSetAttribute(mma_bench5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    auto out = torch::zeros({grid}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    mma_bench5<<<static_cast<int>(grid), 128, smem, at::cuda::getCurrentCUDAStream()>>>(A5, static_cast<int>(n), static_cast<int>(rounds), static_cast<int>(cp),
                                                                                       static_cast<int>(tma), reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()),
                                                                                       static_cast<int>(accm));
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
torch::Tensor run4(int64_t pair, int64_t per, int64_t n, int64_t rounds, int64_t depth, int64_t grid) {
    auto out = torch::zeros({grid}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto* o = reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>());
    const int nn = static_cast<int>(n), rr = static_cast<int>(rounds), dd = static_cast<int>(depth), gg = static_cast<int>(grid);
    if (pair) {
        if (per == 8) launch4<8, true>(nn, rr, dd, gg, o); else if (per == 16) launch4<16, true>(nn, rr, dd, gg, o); else launch4<32, true>(nn, rr, dd, gg, o);
    } else {
        if (per == 8) launch4<8, false>(nn, rr, dd, gg, o); else if (per == 16) launch4<16, false>(nn, rr, dd, gg, o); else launch4<32, false>(nn, rr, dd, gg, o);
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
torch::Tensor run3(int64_t pair, int64_t n, int64_t per, int64_t rounds, int64_t cper, int64_t mode, int64_t grid) {
    auto out = torch::zeros({grid}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    if (pair) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(static_cast<int>(grid)); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = SMEM; cfg.stream = at::cuda::getCurrentCUDAStream();
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        cfg.attrs = at; cfg.numAttrs = 1;
        cudaFuncSetAttribute(mma_bench3<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaLaunchKernelEx(&cfg, mma_bench3<true>, static_cast<int>(n), static_cast<int>(per), static_cast<int>(rounds), static_cast<int>(cper),
                           static_cast<int>(mode), reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    } else {
        cudaFuncSetAttribute(mma_bench3<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        mma_bench3<false><<<static_cast<int>(grid), 128, SMEM, at::cuda::getCurrentCUDAStream()>>>(static_cast<int>(n), static_cast<int>(per),
            static_cast<int>(rounds), static_cast<int>(cper), static_cast<int>(mode), reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
torch::Tensor run(int64_t n, int64_t per, int64_t rounds, int64_t cp, int64_t depth, int64_t atype, int64_t grid, int64_t ilv) {
    TORCH_CHECK(ilv >= 1 && (n > 64 ? (ilv == 1 && per <= 4 && n <= 256) : 2 * ilv * (n <= 32 ? 32 : 64) <= 256));
    auto out = torch::zeros({grid}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaFuncSetAttribute(mma_bench, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    mma_bench<<<static_cast<int>(grid), 128, SMEM, at::cuda::getCurrentCUDAStream()>>>(static_cast<int>(n), static_cast<int>(per), static_cast<int>(rounds),
                                                                                     static_cast<int>(cp), static_cast<int>(depth), static_cast<int>(atype), static_cast<int>(ilv),
                                                                                     reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    return out;
}
#endif
}  // namespace b2mb

#ifndef B2_NO_TORCH
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &b2mb::run); m.def("run2", &b2mb::run2); m.def("run3", &b2mb::run3); m.def("run4", &b2mb::run4); m.def("run5", &b2mb::run5); }
#endif
