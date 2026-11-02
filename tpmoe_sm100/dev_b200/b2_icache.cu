// i-cache retention across launches: a straight-line block of NU unrolled units (~8 instructions each) timed per launch (SM-local clock)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
template <int NU>
__global__ void __launch_bounds__(128, 1) blk(unsigned* io, unsigned long long* out, int salt) {
    unsigned a = threadIdx.x + salt, b = a * 3u + 1u, c = a ^ 0x9e3779b9u, d = b + 7u;
    const unsigned long long t0 = clock64();
#pragma unroll
    for (int i = 0; i < NU; ++i) {
        a = a * 1664525u + (b ^ static_cast<unsigned>(i));
        b = (b >> 3) + c * static_cast<unsigned>(2 * i + 1);
        c = c ^ (a + d);
        d = d + (c >> 7) + static_cast<unsigned>(i);
    }
    const unsigned long long t1 = clock64();
    if (threadIdx.x == 0) { out[blockIdx.x] = t1 - t0; }
    if (a + b + c + d == 0x12345678u) io[0] = a;
}
torch::Tensor run(int64_t nu, int64_t salt) {
    auto out = torch::zeros({148}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto io = torch::zeros({4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto o = reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>());
    auto p = reinterpret_cast<unsigned*>(io.data_ptr<int>());
    auto st = at::cuda::getCurrentCUDAStream();
    switch (nu) {
        case 256: blk<256><<<148, 128, 0, st>>>(p, o, static_cast<int>(salt)); break;
        case 1024: blk<1024><<<148, 128, 0, st>>>(p, o, static_cast<int>(salt)); break;
        case 3072: blk<3072><<<148, 128, 0, st>>>(p, o, static_cast<int>(salt)); break;
        case 6144: blk<6144><<<148, 128, 0, st>>>(p, o, static_cast<int>(salt)); break;
        default: blk<12288><<<148, 128, 0, st>>>(p, o, static_cast<int>(salt)); break;
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
