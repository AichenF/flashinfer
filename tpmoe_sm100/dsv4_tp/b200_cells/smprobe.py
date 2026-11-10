"""smprobe.py: SM availability / co-residency probe.  One CTA per SM (384 threads, ~200 KB dynamic smem, so at most one per SM), each a
fixed dependent FMA chain; per CTA: smid, start / end (globaltimer ns), clock cycles.  A free GPU: every SM once, start spread ~1-2 us,
cycle counts equal.  An SM held by another resident kernel shows up as a late start (smem / warps not available) or as a longer cycle
count (issue slots shared).  probe(dev) -> dict; fmt(d) -> one line."""
import os
import torch
from torch.utils.cpp_extension import load_inline

_CU = r'''
#include <cuda_runtime.h>
#include <torch/extension.h>
__global__ void probe_k(long long* out, int iters) {
    extern __shared__ char sm[];
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    const long long c0 = clock64();
    float a = threadIdx.x * 1e-3f, b = 1.0001f;
    for (int i = 0; i < iters; ++i) { a = fmaf(a, b, 0.5f); b = fmaf(b, 0.999f, a * 1e-9f); }
    sm[threadIdx.x] = static_cast<char>(a + b);
    __syncthreads();
    const long long c1 = clock64();
    unsigned long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0) { long long* o = out + blockIdx.x * 4; o[0] = smid; o[1] = static_cast<long long>(t0); o[2] = static_cast<long long>(t1); o[3] = c1 - c0 + sm[5]; }
}
__global__ void hog_k(volatile int* flag, int busy) {   // (H1 test) resident spinner: one warp per CTA, spins until *flag != 0
    float a = threadIdx.x;
    while (*flag == 0) { if (busy) { for (int i = 0; i < 64; ++i) a = fmaf(a, 1.0001f, 0.5f); } else { __nanosleep(1000); } }
    if (a == 12345.f) flag[1] = 1;
}
void hog(torch::Tensor flag, int64_t ctas, int64_t busy) {
    hog_k<<<static_cast<unsigned>(ctas), 32, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<volatile int*>(flag.data_ptr<int>()), static_cast<int>(busy));
}
__global__ void nvl_k(uint4* dst, long long n16) {   // (NVLink probe) 16-B relaxed.sys stores into a peer buffer, grid-strided
    const uint4 v = make_uint4(threadIdx.x, blockIdx.x, 1u, 2u);
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n16; i += static_cast<long long>(gridDim.x) * blockDim.x)
        asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(dst + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
void nvl(int64_t dst, int64_t nbytes, int64_t grid) {
    nvl_k<<<static_cast<unsigned>(grid), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<uint4*>(dst), nbytes / 16);
}
__global__ void lat_k(const unsigned* base, long long nlines, int nld, unsigned long long* out) {   // (latency probe) 32 dependent random-load chains
    unsigned long long x = 0x9E3779B97F4A7C15ull * (threadIdx.x + 1);
    unsigned acc = 0;
    unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (int i = 0; i < nld; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull + (acc & 1u);
        const long long line = static_cast<long long>((x >> 17) % static_cast<unsigned long long>(nlines));
        acc += __ldcg(base + line * 32);
    }
    unsigned long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = acc; }
}
void lat(int64_t ptr, int64_t nbytes, int64_t nld, torch::Tensor out) {
    lat_k<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const unsigned*>(ptr), nbytes / 128, static_cast<int>(nld),
                                                          reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
}
void probe(torch::Tensor out, int64_t grid, int64_t smem, int64_t iters) {
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(probe_k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)); attr = true; }
    probe_k<<<static_cast<unsigned>(grid), 384, static_cast<size_t>(smem), at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<long long*>(out.data_ptr<int64_t>()), static_cast<int>(iters));
}
'''
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        bd = os.path.join(os.environ.get('SB_BUILD', '/tmp'), 'smprobe')
        os.makedirs(bd, exist_ok=True)
        _EXT = load_inline('smprobe_ext', cpp_sources='void probe(torch::Tensor out, int64_t grid, int64_t smem, int64_t iters); void hog(torch::Tensor flag, int64_t ctas, int64_t busy); void nvl(int64_t dst, int64_t nbytes, int64_t grid); void lat(int64_t ptr, int64_t nbytes, int64_t nld, torch::Tensor out);',
                           cuda_sources=_CU.replace('#include <torch/extension.h>', '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>'),
                           functions=['probe', 'hog', 'nvl', 'lat'], build_directory=bd, extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a'], verbose=False)
    return _EXT


def probe(dev, smem=200 * 1024, iters=20000):
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    out = torch.zeros(G * 4, dtype=torch.int64, device=dev)
    _ext().probe(out, G, smem, iters)
    torch.cuda.synchronize(dev)
    o = out.view(G, 4).cpu()
    sm, t0, t1, cyc = o[:, 0], o[:, 1], o[:, 2], o[:, 3].double()
    start = (t0 - t0.min()).double() / 1e3
    med = cyc.median().item()
    slow = [(int(sm[i]), round(cyc[i].item() / med, 2), round(start[i].item(), 1)) for i in range(G) if cyc[i] > 1.05 * med or start[i] > 5.0]
    return dict(nsm=len(set(sm.tolist())), G=G, start_max=start.max().item(), cyc_ratio=(cyc.max() / med).item(), dur_us=((t1 - t0).double().median() / 1e3).item(), slow=slow[:6])


def fmt(d):
    return f"sm {d['nsm']}/{d['G']} start_max {d['start_max']:.1f} us cyc max/med {d['cyc_ratio']:.2f} dur {d['dur_us']:.1f} us" + (f" slow {d['slow']}" if d['slow'] else '')


_HOG = None


def hog_start(dev, ctas, busy=1):
    """(H1 test) launch a resident spinner of `ctas` one-warp CTAs on a side stream (returns immediately)"""
    global _HOG
    flag = torch.zeros(2, dtype=torch.int32, device=dev)
    st = torch.cuda.Stream(device=dev)
    with torch.cuda.stream(st):
        _ext().hog(flag, ctas, busy)
    _HOG = (flag, st)


def hog_stop():
    global _HOG
    if _HOG is None:
        return
    flag, st = _HOG
    flag[0:1].fill_(1)   # (on the current stream: runs while the spinner is resident)
    st.synchronize()
    _HOG = None


def bw(dev, peer_ptr, nbytes=32 << 20, hbm_bytes=512 << 20):
    """(link / memory probes) GB/s of 16-B relaxed.sys stores into a peer buffer (all SMs), and of a device-local copy"""
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    _ext().nvl(peer_ptr, nbytes, G)   # (warm)
    e0.record(); _ext().nvl(peer_ptr, nbytes, G); e1.record(); torch.cuda.synchronize(dev)
    nv = nbytes / (e0.elapsed_time(e1) * 1e-3) / 1e9
    a = torch.empty(hbm_bytes, dtype=torch.uint8, device=dev); b = torch.empty_like(a)
    b.copy_(a)
    e0.record(); b.copy_(a); e1.record(); torch.cuda.synchronize(dev)
    hb = 2 * hbm_bytes / (e0.elapsed_time(e1) * 1e-3) / 1e9
    del a, b
    return nv, hb


def lat(dev, ptr, nbytes, nld=1000):
    """(latency probe) ns per dependent random 4-B L2-cg load over [ptr, ptr + nbytes) (32 chains): TLB / page-size sensitive"""
    o = torch.zeros(2, dtype=torch.int64, device=dev)
    _ext().lat(ptr, nbytes, 100, o)   # (warm the code)
    _ext().lat(ptr, nbytes, nld, o)
    torch.cuda.synchronize(dev)
    return o[0].item() / nld


def lat_l2(dev, buf, nld=2000):
    """(L2-hit latency probe) the buffer read once (L2-resident if it fits), then ns per dependent random 4-B cg load within it"""
    torch.cuda.synchronize(dev)
    buf.view(torch.int32).sum()   # (touch every line)
    torch.cuda.synchronize(dev)
    return lat(dev, buf.data_ptr(), buf.numel() * buf.element_size(), nld)
