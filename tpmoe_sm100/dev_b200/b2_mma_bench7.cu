// B200 microbenchmark: tcgen05.mma kind::mxf8f6f4.block_scale.block32 cta_group::1, M128 x N x K32 -- the tpmoe_b2 FC1 shape
// (A = weights e2m1 128 rows, B = tokens e4m3 N rows) vs the swapped shape (A = tokens e4m3 128 rows padded, B = weights e2m1 N rows).
// Templated per shape: descriptors computed once, inner loop fully unrolled (check the SASS for R2UR in the issue loop).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace b2m7 {

__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t sbo) {
    return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>(1) << 16) | (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) |
           (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
                 " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}"
                 ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, uint64_t desc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(desc) : "memory"); }
__device__ __forceinline__ bool elect_one() {
    uint32_t pred = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ void tc_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

// smem: A 2 chunks x 16 KB (128 rows x 128 B, SW128), B 2 chunks x N x 128 B, SF 4 x 512 B, bars
template <int N, int PER, int SWAP>
__global__ void __launch_bounds__(128, 1) mb7(int rounds, int depth, unsigned long long* out) {
    constexpr int A_OFF = 0, B_OFF = 32768, SF_OFF = B_OFF + 2 * N * 128, BAR_OFF = SF_OFF + 2048;
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
    uint8_t* base = smem_raw + (sb - sraw);
    __shared__ uint32_t tslot;
    const int tid = threadIdx.x, warp = tid >> 5;
    for (int i = tid; i < BAR_OFF / 4; i += 128) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        reinterpret_cast<uint32_t*>(base)[i] = i * 4 >= SF_OFF ? 0x7F7F7F7Fu : (h & 0x37373737u);
    }
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) mbar_init(sb + BAR_OFF + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem = __shfl_sync(0xffffffffu, tslot, 0);
    if (warp == 1) {
        // A type bits 7-9, B type bits 10-12 (e4m3 0, e2m1 5), N >> 3 at 17, UE8M0 scales (bit 23), M >> 4 at 24
        constexpr uint32_t AT = SWAP ? 0u : 5u, BT = SWAP ? 5u : 0u;
        constexpr uint32_t ID = (AT << 7) | (BT << 10) | (static_cast<uint32_t>(N >> 3) << 17) | (1u << 23) | ((128u >> 4) << 24);
        const uint64_t ad = smem_desc(sb + A_OFF, 1024), bd = smem_desc(sb + B_OFF, 1024);
        const uint32_t sfa = tmem + 256, sfb = tmem + 384;
        unsigned long long t0 = 0;
        for (int r = 0; r < rounds + 8; ++r) {
            if (r == 8) t0 = clock64();
            if (r >= depth) mbar_wait(sb + BAR_OFF + 8 * ((r - depth) & 7), ((r - depth) >> 3) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            if (elect_one()) {
                tc_cp_sf(sfa, smem_desc(sb + SF_OFF, 128) & ~(3ull << 61));
                tc_cp_sf(sfb, smem_desc(sb + SF_OFF + 512, 128) & ~(3ull << 61));
                if (N > 128) tc_cp_sf(sfb + 4, smem_desc(sb + SF_OFF + 1024, 128) & ~(3ull << 61));
#pragma unroll
                for (int j = 0; j < PER; ++j) {
                    const int c = (j >> 2) & 1, kb = j & 3;
                    tc_mma(tmem, ad + c * (16384 >> 4) + 2 * kb, bd + c * ((N * 128) >> 4) + 2 * kb, ID | (kb << 4) | (kb << 29), sfa, sfb, (r | j) != 0);
                }
                tc_commit(sb + BAR_OFF + 8 * (r & 7));
            }
            __syncwarp();
        }
        for (int r = rounds + 8 - depth; r < rounds + 8; ++r) if (r >= 0) mbar_wait(sb + BAR_OFF + 8 * (r & 7), (r >> 3) & 1);
        const unsigned long long t1 = clock64();
        if (tid == 32) out[blockIdx.x] = t1 - t0;
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(512) : "memory");
    }
}

template <int N, int PER, int SWAP>
torch::Tensor go(int rounds, int depth, int grid) {
    constexpr int SMEM = 32768 + 2 * N * 128 + 2048 + 64 + 1024;
    auto out = torch::zeros({grid}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaFuncSetAttribute(mb7<N, PER, SWAP>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    mb7<N, PER, SWAP><<<grid, 128, SMEM, at::cuda::getCurrentCUDAStream()>>>(rounds, depth, reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    return out;
}

torch::Tensor run(int64_t shape, int64_t rounds, int64_t depth, int64_t grid) {
    switch (shape) {
        case 0: return go<8, 8, 0>(rounds, depth, grid);
        case 1: return go<32, 8, 0>(rounds, depth, grid);
        case 2: return go<8, 4, 0>(rounds, depth, grid);
        case 3: return go<128, 8, 1>(rounds, depth, grid);
        case 4: return go<256, 8, 1>(rounds, depth, grid);
        case 5: return go<256, 4, 1>(rounds, depth, grid);
        case 6: return go<128, 8, 0>(rounds, depth, grid);
    }
    TORCH_CHECK(false, "shape");
}
}  // namespace b2m7

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &b2m7::run); }
