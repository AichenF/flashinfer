// Shared-expert numerics of the production DeepSeek-V4 decode path (sglang fork, H20), instruction for instruction:
//   x -> per_token_group_quant_8bit_v2 (group 128, -use_fast_math) -> deep_gemm sm90 fp8 1d2d (per-128 promotion) -> BF16 gate_up
//   -> silu_and_mul_clamp (-use_fast_math, BF16 clamp, BF16 out) -> group quant -> deep_gemm -> BF16 shared output.
// The fast-math SASS was read from the production JIT binaries (cuobjdump): MUFU.RCP / MUFU.EX2 with FTZ multiplies.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>

namespace shnum {

__device__ __forceinline__ float rcp_approx_ftz(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float ex2_approx_ftz(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float mul_ftz(float a, float b) { float r; asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float add_ftz(float a, float b) { float r; asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float max_ftz(float a, float b) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float min_ftz(float a, float b) { float r; asm("min.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float absmax_ftz(float m, float v) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(m), "f"(fabsf(v))); return r; }

// group quant (v2, group 128, fp8 e4m3, float scales): amax = max(1e-10, max |x|); multiplier = 448 * rcp(amax);
// stored scale (the GEMM's dequant factor) = amax * fp32(1/448); q = RN satfinite e4m3(clamp(x * multiplier, -448, 448))
__device__ __forceinline__ float gq_amax_init() { return 1.00000001335143196e-10f; }
__device__ __forceinline__ float gq_mult(float amax) { return mul_ftz(rcp_approx_ftz(amax), 448.0f); }
__device__ __forceinline__ float gq_scale(float amax) { return mul_ftz(amax, 0.0022321429569274187088f); }
__device__ __forceinline__ uint16_t gq_pack2(float a, float b, float mult) {
    float qa = min_ftz(max_ftz(mul_ftz(a, mult), -448.0f), 448.0f);
    float qb = min_ftz(max_ftz(mul_ftz(b, mult), -448.0f), 448.0f);
    uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(qb), "f"(qa)); return r;
}

// silu_and_mul_clamp (limit 10): gate = min(gate, 10), up = clamp(up, -10, 10) in BF16; silu(g) = g * rcp(1 + ex2(g * -log2 e));
// out = BF16_rn(u * silu(g)) -- FTZ throughout
__device__ __forceinline__ __nv_bfloat162 silu_clamp2(__nv_bfloat162 g, __nv_bfloat162 u) {
    const __nv_bfloat162 lim = __floats2bfloat162_rn(10.0f, 10.0f), nlim = __floats2bfloat162_rn(-10.0f, -10.0f);
    g = __hmin2(g, lim);
    u = __hmin2(__hmax2(u, nlim), lim);
    const float g0 = __low2float(g), g1 = __high2float(g), u0 = __low2float(u), u1 = __high2float(u);
    const float s0 = mul_ftz(g0, rcp_approx_ftz(add_ftz(ex2_approx_ftz(mul_ftz(g0, -1.4426950216293334961f)), 1.0f)));
    const float s1 = mul_ftz(g1, rcp_approx_ftz(add_ftz(ex2_approx_ftz(mul_ftz(g1, -1.4426950216293334961f)), 1.0f)));
    return __floats2bfloat162_rn(mul_ftz(u0, s0), mul_ftz(u1, s1));
}

}  // namespace shnum
