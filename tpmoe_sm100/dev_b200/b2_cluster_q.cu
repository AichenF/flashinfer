// max active clusters on this B200 for a 1-CTA/SM kernel with ~227 KB of dynamic smem, cluster sizes 1/2/4/8
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void dummy_k(int* p) { extern __shared__ int s[]; if (threadIdx.x == 0 && p) p[blockIdx.x] = s[0]; }
std::vector<int64_t> query(int64_t smem) {
    std::vector<int64_t> r;
    cudaFuncSetAttribute(dummy_k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    cudaFuncSetAttribute(dummy_k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    for (int cs : {1, 2, 4, 8, 16}) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(cs * 148); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = static_cast<size_t>(smem);
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        cfg.attrs = at; cfg.numAttrs = 1;
        int n = -1;
        cudaError_t e = cudaOccupancyMaxActiveClusters(&n, dummy_k, &cfg);
        r.push_back(e == cudaSuccess ? n : -static_cast<int64_t>(e));
    }
    return r;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("query", &query); }
