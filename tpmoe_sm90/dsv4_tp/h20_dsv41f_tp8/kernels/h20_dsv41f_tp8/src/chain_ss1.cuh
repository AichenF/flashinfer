// AUTO-GENERATED (v41): one SS e4m3 wgmma m64nNk32, C = A(da) * B(db) (scale-d 0)
#pragma once
#include <cstdint>

namespace chain_ss {

template <int N> struct ChainSS1;

template <> struct ChainSS1<8> {
    __device__ __forceinline__ static void run(float (&C)[4], uint64_t da, uint64_t db) {
        asm volatile("{\n"
            ".reg .pred p0;\n"
            "setp.ne.b32 p0, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, p0, 1, 1;\n"
            "}\n"
            : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]) : "l"(da), "l"(db));
    }
};

template <> struct ChainSS1<16> {
    __device__ __forceinline__ static void run(float (&C)[8], uint64_t da, uint64_t db) {
        asm volatile("{\n"
            ".reg .pred p0;\n"
            "setp.ne.b32 p0, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p0, 1, 1;\n"
            "}\n"
            : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7]) : "l"(da), "l"(db));
    }
};

template <> struct ChainSS1<24> {
    __device__ __forceinline__ static void run(float (&C)[12], uint64_t da, uint64_t db) {
        asm volatile("{\n"
            ".reg .pred p0;\n"
            "setp.ne.b32 p0, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, %12, %13, p0, 1, 1;\n"
            "}\n"
            : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7]), "+f"(C[8]), "+f"(C[9]), "+f"(C[10]), "+f"(C[11]) : "l"(da), "l"(db));
    }
};

template <> struct ChainSS1<32> {
    __device__ __forceinline__ static void run(float (&C)[16], uint64_t da, uint64_t db) {
        asm volatile("{\n"
            ".reg .pred p0;\n"
            "setp.ne.b32 p0, 0, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p0, 1, 1;\n"
            "}\n"
            : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7]), "+f"(C[8]), "+f"(C[9]), "+f"(C[10]), "+f"(C[11]), "+f"(C[12]), "+f"(C[13]), "+f"(C[14]), "+f"(C[15]) : "l"(da), "l"(db));
    }
};

}  // namespace chain_ss
