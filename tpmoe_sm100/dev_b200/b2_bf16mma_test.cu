// Does a cuBLAS K-part (torch.mm on x[:, k0:k1], w[:, k0:k1]) equal a tcgen05 kind::f16 chain over K in order (K16 per MMA)?
// One CTA per 128-expert tile: A = W tile [128][K-part] bf16 (K-major SW128, 64-element chunks), B = X [N][K-part] (N tokens padded to 16),
// D fp32 in TMEM (lane = expert, column = token) -> out [256][N].
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
namespace bmt {
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint64_t sdesc(uint32_t addr) {
    return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (static_cast<uint64_t>(0) << 16) | (static_cast<uint64_t>((1024 >> 4) & 0x3FFF) << 32) | (1ull << 46) | (2ull << 61);
}
// smem: per 64-element K chunk: A [128 rows][128 B] (16 KB) then B [Np rows][128 B]
__global__ void __launch_bounds__(128, 1) k(const __nv_bfloat16* X, const __nv_bfloat16* W, float* out, int N, int Np, int K, int k0, int k1) {
    extern __shared__ uint8_t smraw[];
    const uint32_t s0 = (su32(smraw) + 1023u) & ~1023u;
    uint8_t* sm = smraw + (s0 - su32(smraw));
    __shared__ uint32_t tslot;
    __shared__ __align__(8) uint64_t bar;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, et = blockIdx.x;   // expert tile
    const int nch = (k1 - k0) / 64;
    if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(su32(&bar))); asm volatile("fence.mbarrier_init.release.cluster;"); }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem = tslot;
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | (static_cast<uint32_t>(Np >> 3) << 17) | ((128u >> 4) << 24);
    for (int c = 0; c < nch; ++c) {   // one 64-element K chunk at a time through the same smem
        for (int i = tid; i < 128 * 8; i += 128) {
            const int r = i >> 3, g = i & 7;
            const uint4 v = *reinterpret_cast<const uint4*>(W + static_cast<size_t>(et * 128 + r) * K + k0 + c * 64 + g * 8);
            *reinterpret_cast<uint4*>(sm + (r >> 3) * 1024 + (r & 7) * 128 + ((g ^ (r & 7)) << 4)) = v;
        }
        for (int i = tid; i < Np * 8; i += 128) {
            const int r = i >> 3, g = i & 7;
            const uint4 v = r < N ? *reinterpret_cast<const uint4*>(X + static_cast<size_t>(r) * K + k0 + c * 64 + g * 8) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(sm + 16384 + (r >> 3) * 1024 + (r & 7) * 128 + ((g ^ (r & 7)) << 4)) = v;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            uint32_t pred = 0;
            asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
            if (pred) {
                const uint64_t ad = sdesc(s0), bd = sdesc(s0 + 16384);
                for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t acc = (c | kb) != 0;
                    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                                 ::"r"(tmem), "l"(ad + 2 * kb), "l"(bd + 2 * kb), "r"(idesc), "r"(acc) : "memory");
                }
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(su32(&bar)) : "memory");
            }
            __syncwarp();
        }
        asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}" ::"r"(su32(&bar)), "r"(c & 1) : "memory");
        __syncthreads();
    }

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    // lane quadrant warp: TMEM lanes 32 warp .. ; columns = tokens
    for (int c0 = 0; c0 < Np; c0 += 8) {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n\ttcgen05.wait::ld.sync.aligned;"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(tmem + ((32u * warp) << 16) + c0) : "memory");
        for (int j = 0; j < 8; ++j) if (c0 + j < N) out[static_cast<size_t>(et * 128 + 32 * warp + lane) * N + c0 + j] = __uint_as_float(r[j]);
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256) : "memory");
    }
}
torch::Tensor run(torch::Tensor x, torch::Tensor w, int64_t k0, int64_t k1) {
    const int N = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1)), Np = (N + 15) & ~15;
    auto out = torch::zeros({256, N}, torch::dtype(torch::kFloat32).device(x.device()));
    const int smem = 16384 + Np * 128 + 1024;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    k<<<2, 128, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
                                                          out.data_ptr<float>(), N, Np, K, static_cast<int>(k0), static_cast<int>(k1));
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

// swapped roles: A = X [128 rows (tokens, zero-padded)][K], B = W [16 experts][K] (N = 16); block eb = experts 16 eb ..; out [256][N]
__global__ void __launch_bounds__(128, 1) ks(const __nv_bfloat16* X, const __nv_bfloat16* W, float* out, int N, int K, int k0, int k1) {
    extern __shared__ uint8_t smraw[];
    const uint32_t s0 = (su32(smraw) + 1023u) & ~1023u;
    uint8_t* sm = smraw + (s0 - su32(smraw));
    __shared__ uint32_t tslot;
    __shared__ __align__(8) uint64_t bar;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, eb = blockIdx.x;
    const int nch = (k1 - k0) / 64;
    if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(su32(&bar))); asm volatile("fence.mbarrier_init.release.cluster;"); }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tslot)), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t tmem = tslot;
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | (static_cast<uint32_t>(16 >> 3) << 17) | ((128u >> 4) << 24);
    for (int c = 0; c < nch; ++c) {
        for (int i = tid; i < 128 * 8; i += 128) {
            const int r = i >> 3, g = i & 7;
            const uint4 v = r < N ? *reinterpret_cast<const uint4*>(X + static_cast<size_t>(r) * K + k0 + c * 64 + g * 8) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(sm + (r >> 3) * 1024 + (r & 7) * 128 + ((g ^ (r & 7)) << 4)) = v;
        }
        {
            const int i = tid, r = i >> 3, g = i & 7;   // 16 rows x 8 granules = 128 threads
            const uint4 v = *reinterpret_cast<const uint4*>(W + static_cast<size_t>(eb * 16 + r) * K + k0 + c * 64 + g * 8);
            *reinterpret_cast<uint4*>(sm + 16384 + (r >> 3) * 1024 + (r & 7) * 128 + ((g ^ (r & 7)) << 4)) = v;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            uint32_t pred = 0;
            asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
            if (pred) {
                const uint64_t ad = sdesc(s0), bd = sdesc(s0 + 16384);
                for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t acc = (c | kb) != 0;
                    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                                 ::"r"(tmem), "l"(ad + 2 * kb), "l"(bd + 2 * kb), "r"(idesc), "r"(acc) : "memory");
                }
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(su32(&bar)) : "memory");
            }
            __syncwarp();
        }
        asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}" ::"r"(su32(&bar)), "r"(c & 1) : "memory");
        __syncthreads();
    }
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\ttcgen05.wait::ld.sync.aligned;"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                   "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(tmem + ((32u * warp) << 16)) : "memory");
    const int t = 32 * warp + lane;   // token row
    if (t < N) for (int j = 0; j < 16; ++j) out[static_cast<size_t>(eb * 16 + j) * N + t] = __uint_as_float(r[j]);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(32) : "memory");
    }
}
// dsv3_router_gemm emulation (M <= 4): expert e by 128 threads: k-iters ki (4) x 8 elements, fma per token, warp xor 16..1, 0 + w0 + .. + w3
__global__ void __launch_bounds__(128, 1) kd(const __nv_bfloat16* X, const __nv_bfloat16* W, float* out, int M) {
    const int e = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float red[4][4];
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ki = 0; ki < 4; ++ki) {
        const int kb = ki * 1024 + tid * 8;
        const uint4 wv = *reinterpret_cast<const uint4*>(W + static_cast<size_t>(e) * 4096 + kb);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
        for (int m = 0; m < 4; ++m) {
            if (m >= M) break;
            const uint4 xv = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * 4096 + kb);
            const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
            for (int k = 0; k < 8; ++k) acc[m] = __fmaf_rn(__bfloat162float(xb[k]), __bfloat162float(wb[k]), acc[m]);
        }
    }
    for (int m = 0; m < 4; ++m) {
        if (m >= M) break;
        float v = acc[m];
        for (int o = 16; o >= 1; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[m][warp] = v;
    }
    __syncthreads();
    if (warp == 0 && lane < M) {
        float f = 0.f;
        for (int w = 0; w < 4; ++w) f += red[lane][w];
        out[static_cast<size_t>(lane) * 256 + e] = f;
    }
}
torch::Tensor run_swap(torch::Tensor x, torch::Tensor w, int64_t k0, int64_t k1) {
    const int N = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
    auto out = torch::zeros({256, N}, torch::dtype(torch::kFloat32).device(x.device()));
    const int smem = 16384 + 2048 + 1024;
    cudaFuncSetAttribute(ks, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    ks<<<16, 128, smem, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
                                                            out.data_ptr<float>(), N, K, static_cast<int>(k0), static_cast<int>(k1));
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
torch::Tensor run_dsv3(torch::Tensor x, torch::Tensor w) {
    const int M = static_cast<int>(x.size(0));
    auto out = torch::zeros({M, 256}, torch::dtype(torch::kFloat32).device(x.device()));
    kd<<<256, 128, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
                                                          out.data_ptr<float>(), M);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &bmt::run); m.def("run_swap", &bmt::run_swap); m.def("run_dsv3", &bmt::run_dsv3); }
