// All-reduce transport micro-benchmark for the tpmoe_b2 tail (B200 TP4, one process driving 4 GPUs):
//   * one-way latency: ping-pong of one 16-B LL message GPU0 <-> GPU1, unicast st.relaxed.sys vs multimem.st (NVLS multicast)
//   * one-shot LL all-reduce of M x 4096 bf16 (43 x 16-B messages per 256 values, 12 B data + epoch tag) with each rank's partial
//     pushed to the 4 ranks by 4 unicast stores (U) or one multimem.st (MC); receivers poll their 4 sources and sum in rank order
//   * RS + AG LL all-reduce: reduce-scatter unicast to the owner of each quarter, owner sums in rank order, all-gather by 4 unicast
//     stores (U) or one multimem.st (MC), then every rank pulls its messages
// Every timed run is checked bitwise against a host reference (bf16(fp32 sum in rank order 0..3)).
// Build: nvcc -O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a b2_ar_bench.cu -lcuda -o b2_ar_bench
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CU(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { const char* s_; cuGetErrorString(e_, &s_); printf("CU %s at %s:%d\n", s_, __FILE__, __LINE__); exit(1); } } while (0)

constexpr int NDEV = 4, DIM = 4096, NRT = DIM / 256, LRT = NRT / NDEV, NMSG = 43, MAXM = 128, THREADS = 256;
constexpr size_t PAR = static_cast<size_t>(NDEV) * NRT * MAXM * NMSG;   // uint4 per parity half (os: [src][rtile][t][m]; rs: [src][lrt][t][m])
constexpr size_t AGP = static_cast<size_t>(NRT) * MAXM * NMSG;          // ag: [rtile][t][m]

__device__ __forceinline__ void st_ll(uint4* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void st_mc(uint4* p, uint4 v) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(__uint_as_float(v.x)), "f"(__uint_as_float(v.y)), "f"(__uint_as_float(v.z)),
                 "f"(__uint_as_float(v.w)) : "memory");
}
__device__ __forceinline__ uint4 ld_ll(const uint4* p) { uint4 v; asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ float bflo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bfhi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

struct Bufs {
    uint4* os[NDEV];    // one-shot receive buffers (2 parity halves), unicast VA of each rank
    uint4* osmc;        // their multicast VA (this device's mapping)
    uint4* ag[NDEV];    // all-gather receive buffers
    uint4* agmc;
};
struct Args {
    Bufs B;
    int rank, M, mc;
    unsigned epoch;
    const __nv_bfloat16* in;   // this rank's partial [M][DIM]
    __nv_bfloat16* out;        // [M][DIM]
};

__device__ __forceinline__ uint4 msg_of(const __nv_bfloat16* row, int m, unsigned epoch) {   // values 6 m .. 6 m + 5 of a 256-row tile
    const uint32_t* w = reinterpret_cast<const uint32_t*>(row + 6 * m);
    return make_uint4(w[0], w[1], m < NMSG - 1 ? w[2] : 0u, epoch);
}
__device__ __forceinline__ void put_out(__nv_bfloat16* row, int m, uint4 v) {
    uint32_t* o = reinterpret_cast<uint32_t*>(row + 6 * m);
    o[0] = v.x; o[1] = v.y;
    if (m < NMSG - 1) o[2] = v.z;
}
__device__ __forceinline__ uint4 sum4(const uint4* v, unsigned epoch) {
    float a[6] = {bflo(v[0].x), bfhi(v[0].x), bflo(v[0].y), bfhi(v[0].y), bflo(v[0].z), bfhi(v[0].z)};
#pragma unroll
    for (int d = 1; d < NDEV; ++d) {
        a[0] = __fadd_rn(a[0], bflo(v[d].x)); a[1] = __fadd_rn(a[1], bfhi(v[d].x));
        a[2] = __fadd_rn(a[2], bflo(v[d].y)); a[3] = __fadd_rn(a[3], bfhi(v[d].y));
        a[4] = __fadd_rn(a[4], bflo(v[d].z)); a[5] = __fadd_rn(a[5], bfhi(v[d].z));
    }
    return make_uint4(pack2(a[0], a[1]), pack2(a[2], a[3]), pack2(a[4], a[5]), epoch);
}
__device__ __forceinline__ void poll4(const uint4* base, size_t stride, unsigned epoch, uint4* v) {
    unsigned pend = (1u << NDEV) - 1u;
    while (pend) {
        const unsigned q = pend;
#pragma unroll
        for (int d = 0; d < NDEV; ++d) if (q & (1u << d)) v[d] = ld_ll(base + d * stride);
#pragma unroll
        for (int d = 0; d < NDEV; ++d) if ((q & (1u << d)) && v[d].w == epoch) pend &= ~(1u << d);
    }
}

// one-shot: message x = (t, rtile, m) of this rank's partial -> slot [rank][rtile][t][m] of every rank; then receive this CTA's share
__global__ void __launch_bounds__(THREADS) k_oneshot(Args A) {
    const size_t par = (A.epoch & 1u) * PAR, src = static_cast<size_t>(NRT) * MAXM * NMSG;
    const int tot = A.M * NRT * NMSG, G = gridDim.x;
    for (int x = blockIdx.x * THREADS + threadIdx.x; x < tot; x += G * THREADS) {
        const int p = x / NMSG, m = x - p * NMSG, t = p / NRT, rt = p - t * NRT;
        const uint4 v = msg_of(A.in + static_cast<size_t>(t) * DIM + rt * 256, m, A.epoch);
        const size_t i = par + A.rank * src + (static_cast<size_t>(rt) * MAXM + t) * NMSG + m;
        if (A.mc) st_mc(A.B.osmc + i, v);
        else {
#pragma unroll
            for (int d = 0; d < NDEV; ++d) st_ll(A.B.os[(A.rank + d) & (NDEV - 1)] + i, v);
        }
    }
    // receive: pair j = rtile * M + t over CTAs (as the kernel's one-shot tail)
    const int npair = NRT * A.M, b = blockIdx.x, mine = b < npair ? (npair - b + G - 1) / G : 0;
    const uint4* me = A.B.os[A.rank] + par;
    for (int x = threadIdx.x; x < mine * NMSG; x += THREADS) {
        const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rt = j / A.M, t = j - rt * A.M;
        uint4 v[NDEV];
        poll4(me + (static_cast<size_t>(rt) * MAXM + t) * NMSG + m, src, A.epoch, v);
        put_out(A.out + static_cast<size_t>(t) * DIM + rt * 256, m, sum4(v, A.epoch));
    }
}

// RS + AG: message (t, rtile, m) -> owner rtile / LRT slot [rank][lrt][t][m]; owner T2 pairs (lrt, t) over CTAs; T3 pairs (rtile, t)
__global__ void __launch_bounds__(THREADS) k_rsag(Args A) {
    const size_t par = (A.epoch & 1u) * PAR, src = static_cast<size_t>(LRT) * MAXM * NMSG, agp = (A.epoch & 1u) * AGP;
    const int tot = A.M * NRT * NMSG, G = gridDim.x, b = blockIdx.x;
    for (int x = blockIdx.x * THREADS + threadIdx.x; x < tot; x += G * THREADS) {
        const int p = x / NMSG, m = x - p * NMSG, t = p / NRT, rt = p - t * NRT;
        const uint4 v = msg_of(A.in + static_cast<size_t>(t) * DIM + rt * 256, m, A.epoch);
        st_ll(A.B.os[rt / LRT] + par + A.rank * src + (static_cast<size_t>(rt % LRT) * MAXM + t) * NMSG + m, v);
    }
    {
        const int npair = LRT * A.M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* me = A.B.os[A.rank] + par;
        for (int x = threadIdx.x; x < mine * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, lrt = j / A.M, t = j - lrt * A.M;
            uint4 v[NDEV];
            poll4(me + (static_cast<size_t>(lrt) * MAXM + t) * NMSG + m, src, A.epoch, v);
            const uint4 mo = sum4(v, A.epoch);
            const size_t i = agp + (static_cast<size_t>(A.rank * LRT + lrt) * MAXM + t) * NMSG + m;
            if (A.mc) st_mc(A.B.agmc + i, mo);
            else {
#pragma unroll
                for (int d = 0; d < NDEV; ++d) st_ll(A.B.ag[(A.rank + d) & (NDEV - 1)] + i, mo);
            }
        }
    }
    {
        const int npair = NRT * A.M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* me = A.B.ag[A.rank] + agp;
        for (int x = threadIdx.x; x < mine * NMSG; x += THREADS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rt = j / A.M, t = j - rt * A.M;
            uint4 v;
            do { v = ld_ll(me + (static_cast<size_t>(rt) * MAXM + t) * NMSG + m); } while (v.w != A.epoch);
            put_out(A.out + static_cast<size_t>(t) * DIM + rt * 256, m, v);
        }
    }
}

__global__ void k_empty(Args A) { (void)A; }
// send phase only (no receive): kernel end = last store + whatever the end-of-kernel flush waits for. mode 0 unicast x4, 1 multicast,
// 2 local only (own buffer), 3 one remote peer only
__global__ void __launch_bounds__(THREADS) k_sendonly(Args A, int mode) {
    const size_t par = (A.epoch & 1u) * PAR, src = static_cast<size_t>(NRT) * MAXM * NMSG;
    const int tot = A.M * NRT * NMSG, G = gridDim.x;
    for (int x = blockIdx.x * THREADS + threadIdx.x; x < tot; x += G * THREADS) {
        const int p = x / NMSG, m = x - p * NMSG, t = p / NRT, rt = p - t * NRT;
        const uint4 v = msg_of(A.in + static_cast<size_t>(t) * DIM + rt * 256, m, A.epoch);
        const size_t i = par + A.rank * src + (static_cast<size_t>(rt) * MAXM + t) * NMSG + m;
        if (mode == 1) st_mc(A.B.osmc + i, v);
        else if (mode == 2) st_ll(A.B.os[A.rank] + i, v);
        else if (mode == 3) st_ll(A.B.os[(A.rank + 1) & (NDEV - 1)] + i, v);
        else {
#pragma unroll
            for (int d = 0; d < NDEV; ++d) st_ll(A.B.os[(A.rank + d) & (NDEV - 1)] + i, v);
        }
    }
}

// ping-pong: GPU0 thread 0 sends i to GPU1 (unicast or multicast slot 0), GPU1 echoes i to GPU0's slot 1 (unicast)
__global__ void k_ping(uint4* peer_or_mc, const uint4* mine, int mc, int n, unsigned base, unsigned long long* dt) {
    const unsigned long long t0 = gtime();
    for (int i = 1; i <= n; ++i) {
        const uint4 v = make_uint4(i, i, i, base + i);
        if (mc) st_mc(peer_or_mc, v); else st_ll(peer_or_mc, v);
        while (ld_ll(mine + 1).w != base + i) {}
    }
    *dt = gtime() - t0;
}
__global__ void k_pong(uint4* peer, const uint4* mine, int n, unsigned base) {
    for (int i = 1; i <= n; ++i) {
        while (ld_ll(mine).w != base + i) {}
        st_ll(peer + 1, make_uint4(i, i, i, base + i));
    }
}

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return static_cast<uint16_t>(u >> 16); }
static float bf2f(uint16_t h) { uint32_t u = static_cast<uint32_t>(h) << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    CU(cuInit(0));
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < NDEV) { printf("need %d GPUs\n", NDEV); return 1; }
    CUdevice dev[NDEV];
    int mcsup = 1;
    for (int r = 0; r < NDEV; ++r) {
        CU(cuDeviceGet(&dev[r], r));
        int s = 0;
        CU(cuDeviceGetAttribute(&s, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev[r]));
        mcsup &= s;
        CK(cudaSetDevice(r));
        for (int q = 0; q < NDEV; ++q) if (q != r) { cudaError_t e = cudaDeviceEnablePeerAccess(q, 0); if (e != cudaErrorPeerAccessAlreadyEnabled) CK(e); }
    }
    printf("multicast supported: %d\n", mcsup);
    const size_t osz = 2 * PAR * 16, agsz = 2 * AGP * 16;
    Bufs B[NDEV];
    // buffers: physical allocations bound to a multicast object when supported (unicast VA mapped on every device), else cudaMalloc
    auto make = [&](size_t sz, uint4** uni, uint4** mcva) {
        if (!mcsup) {
            for (int r = 0; r < NDEV; ++r) { CK(cudaSetDevice(r)); void* p; CK(cudaMalloc(&p, sz)); CK(cudaMemset(p, 0, sz)); uni[r] = static_cast<uint4*>(p); mcva[r] = nullptr; }
            return;
        }
        CUmulticastObjectProp mp = {};
        mp.numDevices = NDEV; mp.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
        size_t gran = 0;
        mp.size = sz;
        CU(cuMulticastGetGranularity(&gran, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
        const size_t asz = (sz + gran - 1) / gran * gran;
        mp.size = asz;
        CUmemGenericAllocationHandle mch;
        CU(cuMulticastCreate(&mch, &mp));
        for (int r = 0; r < NDEV; ++r) CU(cuMulticastAddDevice(mch, dev[r]));
        for (int r = 0; r < NDEV; ++r) {
            CK(cudaSetDevice(r));
            CUmemAllocationProp ap = {};
            ap.type = CU_MEM_ALLOCATION_TYPE_PINNED; ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ap.location.id = r;
            size_t g2 = 0;
            CU(cuMemGetAllocationGranularity(&g2, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
            const size_t psz = (asz + g2 - 1) / g2 * g2;
            CUmemGenericAllocationHandle ph;
            CU(cuMemCreate(&ph, psz, &ap, 0));
            CU(cuMulticastBindMem(mch, 0, ph, 0, asz, 0));
            CUdeviceptr va;
            CU(cuMemAddressReserve(&va, psz, 0, 0, 0));
            CU(cuMemMap(va, psz, 0, ph, 0));
            CUmemAccessDesc ad[NDEV];
            for (int q = 0; q < NDEV; ++q) { ad[q].location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad[q].location.id = q; ad[q].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
            CU(cuMemSetAccess(va, psz, ad, NDEV));
            CK(cudaMemset(reinterpret_cast<void*>(va), 0, asz));
            uni[r] = reinterpret_cast<uint4*>(va);
            CUdeviceptr mva;
            CU(cuMemAddressReserve(&mva, asz, 0, 0, 0));
            CU(cuMemMap(mva, asz, 0, mch, 0));
            CUmemAccessDesc md; md.location.type = CU_MEM_LOCATION_TYPE_DEVICE; md.location.id = r; md.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU(cuMemSetAccess(mva, asz, &md, 1));
            mcva[r] = reinterpret_cast<uint4*>(mva);
        }
    };
    uint4 *osu[NDEV], *osm[NDEV], *agu[NDEV], *agm[NDEV];
    make(osz, osu, osm);
    make(agsz, agu, agm);
    for (int r = 0; r < NDEV; ++r) {
        for (int q = 0; q < NDEV; ++q) { B[r].os[q] = osu[q]; B[r].ag[q] = agu[q]; }
        B[r].osmc = osm[r]; B[r].agmc = agm[r];
    }
    for (int r = 0; r < NDEV; ++r) { CK(cudaSetDevice(r)); CK(cudaDeviceSynchronize()); }

    // ---- ping-pong latency (unicast, then multicast)
    {
        unsigned long long* dt;
        CK(cudaSetDevice(0)); CK(cudaMallocManaged(&dt, 8));
        static unsigned base = 1000;
        for (int mc = 0; mc <= mcsup; ++mc) {
            const int n = 2000;
            CK(cudaSetDevice(1)); k_pong<<<1, 1>>>(B[1].os[0], B[1].os[1], n, base);
            CK(cudaSetDevice(0)); k_ping<<<1, 1>>>(mc ? B[0].osmc : B[0].os[1], B[0].os[0], mc, n, base, dt);
            for (int r = 0; r < 2; ++r) { CK(cudaSetDevice(r)); CK(cudaDeviceSynchronize()); }
            printf("ping-pong %s: one-way %.3f us (round trip %.3f us, %d trips)\n", mc ? "multicast" : "unicast  ", *dt / 1e3 / n / 2, *dt / 1e3 / n, n);
            base += 10 * n;
        }
    }

    // ---- all-reduces
    const int Ms[] = {1, 8, 16, 32, 48, 64, 96, 128};
    std::vector<uint16_t> hin[NDEV];
    __nv_bfloat16 *din[NDEV], *dout[NDEV];
    cudaStream_t st[NDEV];
    cudaEvent_t e0[NDEV], e1[NDEV];
    for (int r = 0; r < NDEV; ++r) {
        CK(cudaSetDevice(r));
        CK(cudaMalloc(&din[r], static_cast<size_t>(MAXM) * DIM * 2)); CK(cudaMalloc(&dout[r], static_cast<size_t>(MAXM) * DIM * 2));
        hin[r].resize(static_cast<size_t>(MAXM) * DIM);
        srand(17 + r);
        for (auto& h : hin[r]) h = f2bf((rand() / (float)RAND_MAX - 0.5f) * powf(2.f, static_cast<float>(rand() % 12 - 6)));
        CK(cudaMemcpy(din[r], hin[r].data(), hin[r].size() * 2, cudaMemcpyHostToDevice));
        CK(cudaStreamCreateWithFlags(&st[r], cudaStreamNonBlocking));
        CK(cudaEventCreate(&e0[r])); CK(cudaEventCreate(&e1[r]));
    }
    std::vector<uint16_t> ref(static_cast<size_t>(MAXM) * DIM), got(static_cast<size_t>(MAXM) * DIM);
    for (size_t i = 0; i < ref.size(); ++i) {
        float a = bf2f(hin[0][i]);
        for (int r = 1; r < NDEV; ++r) a = a + bf2f(hin[r][i]);
        ref[i] = f2bf(a);
    }
    unsigned epoch = 1;
    // CUDA graph per device of `iters` calls back to back (epochs epoch .. epoch + iters - 1); proto 2 = empty kernel (launch floor)
    auto timed = [&](int proto, int mc, int M) {
        cudaGraphExec_t ge[NDEV];
        for (int r = 0; r < NDEV; ++r) {
            CK(cudaSetDevice(r));
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(st[r], cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < iters; ++i) {
                Args A{B[r], r, M, mc, epoch + i, din[r], dout[r]};
                if (proto == 0) k_oneshot<<<148, THREADS, 0, st[r]>>>(A);
                else if (proto == 1) k_rsag<<<148, THREADS, 0, st[r]>>>(A);
                else if (proto >= 3) k_sendonly<<<148, THREADS, 0, st[r]>>>(A, proto - 3);
                else k_empty<<<148, THREADS, 0, st[r]>>>(A);
            }
            CK(cudaStreamEndCapture(st[r], &g));
            CK(cudaGraphInstantiate(&ge[r], g, 0));
            CK(cudaGraphDestroy(g));
        }
        for (int r = 0; r < NDEV; ++r) { CK(cudaSetDevice(r)); CK(cudaEventRecord(e0[r], st[r])); CK(cudaGraphLaunch(ge[r], st[r])); CK(cudaEventRecord(e1[r], st[r])); }
        float mx = 0.f;
        for (int r = 0; r < NDEV; ++r) {
            CK(cudaSetDevice(r)); CK(cudaEventSynchronize(e1[r]));
            float ms; CK(cudaEventElapsedTime(&ms, e0[r], e1[r])); mx = fmaxf(mx, ms);
            CK(cudaGraphExecDestroy(ge[r]));
        }
        epoch += iters;
        return mx * 1e3f / iters;
    };
    float fl = 1e30f;
    for (int rep = 0; rep < 3; ++rep) fl = fminf(fl, timed(2, 0, 1));
    printf("empty-kernel graph floor: %.2f us/call\n", fl);
    for (int mode = 0; mode < 4; ++mode) {
        if (mode == 1 && !mcsup) continue;
        for (int M : {1, 8, 32, 128}) {
            float b = 1e30f;
            for (int rep = 0; rep < 3; ++rep) b = fminf(b, timed(3 + mode, 0, M));
            printf("send-only %-10s M %3d: %7.2f us/call (-floor %6.2f)\n", mode == 0 ? "unicast4" : mode == 1 ? "multicast" : mode == 2 ? "local" : "1 peer", M, b, b - fl);
        }
    }
    if (getenv("SENDONLY")) return 0;
    for (int proto = 0; proto < 2; ++proto)
        for (int mc = 0; mc <= mcsup; ++mc)
            for (int M : Ms) {
                float best = 1e30f;
                for (int rep = 0; rep < 3; ++rep) best = fminf(best, timed(proto, mc, M));
                long bad = 0;
                for (int r = 0; r < NDEV; ++r) {
                    CK(cudaSetDevice(r));
                    CK(cudaMemcpy(got.data(), dout[r], static_cast<size_t>(M) * DIM * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < static_cast<size_t>(M) * DIM; ++i) bad += got[i] != ref[i];
                    CK(cudaMemset(dout[r], 0, static_cast<size_t>(MAXM) * DIM * 2));
                }
                printf("%-8s %-9s M %3d: %7.2f us/call (-floor %6.2f; graph of %d, best of 3)  mismatches %ld\n", proto ? "rs+ag" : "one-shot",
                       mc ? "multicast" : "unicast", M, best, best - fl, iters, bad);
            }

    return 0;
}
