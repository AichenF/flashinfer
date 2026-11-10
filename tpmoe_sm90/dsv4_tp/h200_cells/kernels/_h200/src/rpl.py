"""Router GEMM as cuBLAS split-K partials (no splitKreduce kernel): partials(x, w) -> fp32 [S, M, 256] whose fp32 sum (P0 + P1) + P2 is
torch.mm(x, w.t(), out_dtype=fp32) bit for bit (H20, cuBLAS nvjet: S 3 = k [0,1408) [1408,2816) [2816,4096), S 2 = halves at the M
below; every plain-chain tile config gives the same partials, ub/lt_partials.py).  One strided-batched cuBLASLt GEMM: S = 3 batch b
reads the in-bounds x window [1280 b, 1280 b + 1536) against a zero-padded weight copy (zeros outside the split's k range: exact), S = 2
reads the halves of w directly.  The tpmoe kernel (gpl builds) sums the partials in gate_token."""
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>
#define CK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x, " -> ", (int)s_); } while (0)
static cublasLtHandle_t H = nullptr;
struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t al; int64_t key[6]; };
static std::vector<Plan> plans;
static void setb(cublasLtMatrixLayout_t l, int32_t n, int64_t st) {
    CK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &n, sizeof(n)));
    CK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st, sizeof(st)));
}
// S batches: A = weights (col-major Kp x 256, ld lda, batch stride sa, op T), B = x window (col-major Kp x M, ld 4096, batch stride xs),
// C = out [S][M][256]; heuristic algo idx (plain chains: every tile config is bitwise the same)
int64_t run(torch::Tensor wa, torch::Tensor x, torch::Tensor o, torch::Tensor wsb, int64_t S, int64_t Kp, int64_t lda, int64_t sa, int64_t xs, int64_t idx) {
    const int64_t M = x.size(0);
    if (!H) CK(cublasLtCreate(&H));
    Plan* p = nullptr;
    for (auto& q : plans) if (q.key[0] == M && q.key[1] == S && q.key[2] == Kp && q.key[3] == lda && q.key[4] == xs && q.key[5] == idx) p = &q;
    if (!p) {
        Plan q;
        CK(cublasLtMatmulDescCreate(&q.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CK(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CK(cublasLtMatrixLayoutCreate(&q.a, CUDA_R_16BF, Kp, 256, lda));
        CK(cublasLtMatrixLayoutCreate(&q.b, CUDA_R_16BF, Kp, M, 4096));
        CK(cublasLtMatrixLayoutCreate(&q.c, CUDA_R_32F, 256, M, 256));
        setb(q.a, (int32_t)S, sa); setb(q.b, (int32_t)S, xs); setb(q.c, (int32_t)S, 256 * M);
        cublasLtMatmulPreference_t pf; CK(cublasLtMatmulPreferenceCreate(&pf));
        uint64_t w = (uint64_t)wsb.numel(); CK(cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &w, sizeof(w)));
        cublasLtMatmulHeuristicResult_t r[8]; int n = 0;
        CK(cublasLtMatmulAlgoGetHeuristic(H, q.op, q.a, q.b, q.c, q.c, pf, 8, r, &n));
        cublasLtMatmulPreferenceDestroy(pf);
        TORCH_CHECK(idx < n, "rpl: no algo");
        q.al = r[idx].algo;
        q.key[0] = M; q.key[1] = S; q.key[2] = Kp; q.key[3] = lda; q.key[4] = xs; q.key[5] = idx;
        plans.push_back(q); p = &plans.back();
    }
    float alpha = 1.f, beta = 0.f;
    return (int64_t)cublasLtMatmul(H, p->op, &alpha, wa.data_ptr(), p->a, x.data_ptr(), p->b, &beta, o.data_ptr(), p->c, o.data_ptr(), p->c,
                                   &p->al, wsb.data_ptr(), (size_t)wsb.numel(), at::cuda::getCurrentCUDAStream());
}
"""
CPP = "int64_t run(torch::Tensor wa, torch::Tensor x, torch::Tensor o, torch::Tensor wsb, int64_t S, int64_t Kp, int64_t lda, int64_t sa, int64_t xs, int64_t idx);"
_ext = None
S2_H20 = [(97, 103), (121, 127), (201, 207), (209, 224), (241, 271), (288, 288), (313, 319), (337, 351), (353, 360), (432, 448), (480, 504), (512, 512)]
# H100 (cuBLAS nvjet_sm90_tss_*_splitK, tools/rpl_s23.py 10-27): S = 2 in these M ranges, S = 3 elsewhere, and NO exact split (another
# split count / boundary) in S_NONE_H100 -> callers must not use partials there (rpl.exact(M) False)
S2_H100 = [(121, 127), (144, 159), (201, 207), (209, 216), (225, 327), (329, 359), (384, 392), (400, 400), (432, 447), (457, 464), (496, 496), (512, 512)]
S_NONE_H100 = [(328, 328), (360, 383), (393, 399), (401, 408), (416, 431), (449, 456), (465, 495), (497, 511)]
_h100 = None
def _is_h100():
    global _h100
    if _h100 is None:
        import torch
        _h100 = 'H100' in torch.cuda.get_device_name()
    return _h100
def exact(M):   # rpl partials reproduce the served router GEMM bit for bit at this M on this GPU
    return 16 < M <= 512 and not (_is_h100() and any(a <= M <= b for a, b in S_NONE_H100))
S2 = S2_H20
_wp, _ws = {}, {}


def ext():   # built once into ./build_rpl (persistent across the per-run containers)
    global _ext
    if _ext is None:
        import os
        from pathlib import Path
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0a')
        bd = Path(__file__).resolve().parent / 'build_rpl'
        bd.mkdir(exist_ok=True)
        _ext = load_inline('rpl_partials', cpp_sources=[CPP], cuda_sources=[SRC], functions=['run'], extra_ldflags=['-lcublasLt'],
                           build_directory=str(bd), verbose=False)
    return _ext


def splits(M):
    assert 16 < M <= 512, M
    return 2 if any(a <= M <= b for a, b in (S2_H100 if _is_h100() else S2_H20)) else 3


def padded3(w):   # [3][256][1536]: batch b holds w's k range of split b at offset (k - 1280 b), zeros elsewhere
    key = (w.data_ptr(), w.device.index)
    if key not in _wp:
        wp = torch.zeros(3, 256, 1536, dtype=w.dtype, device=w.device)
        for b, (k0, k1) in enumerate([(0, 1408), (1408, 2816), (2816, 4096)]):
            wp[b, :, k0 - 1280 * b:k1 - 1280 * b] = w[:, k0:k1]
        _wp[key] = wp.contiguous()
    return _wp[key]


IDX = {}   # M -> heuristic algo idx (default 0; every heuristic algo is bitwise the same here, winprobe2.py: they differ in SM footprint)


def partials(x, w, idx=None):
    M = x.shape[0]
    if idx is None:
        idx = IDX.get(M, 0)
    S = splits(M)
    E = ext()
    dev = x.device.index
    if dev not in _ws:
        _ws[dev] = torch.empty(4 << 20, dtype=torch.uint8, device=x.device)
    o = torch.empty((S, M, 256), dtype=torch.float32, device=x.device)
    if S == 3:
        st = E.run(padded3(w), x, o, _ws[dev], 3, 1536, 1536, 256 * 1536, 1280, idx)
    else:
        st = E.run(w, x, o, _ws[dev], 2, 2048, 4096, 2048, 2048, idx)
    assert st == 0, st
    return o


def prealloc(w, ms=()):   # the padded copy + workspace up front (model-load time in production); + the cuBLASLt handle / plans of ms
    padded3(w)
    if w.device.index not in _ws:
        _ws[w.device.index] = torch.empty(4 << 20, dtype=torch.uint8, device=w.device)
    for m in ms:
        if 16 < m <= 512:
            partials(torch.zeros((m, w.shape[1]), dtype=w.dtype, device=w.device), w)
