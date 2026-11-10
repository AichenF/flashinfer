"""patch_ro.py: v77 tpmoe_b2.cu -> + TPB2_SPEC=2 (routed-only production config fixed at compile time: TP4, top-6, ids / weights
given, no fused gate, no shared expert).  The routed arithmetic is untouched (bitwise == v77 == production trtllm-gen routed path).
Exact-string patches, each asserted (re-runnable on the pristine v77 source only)."""
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'v77' / 'tpmoe_b2.cu')
s = P.read_text()


def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, f'expected {count}, found {c}: {old[:200]}'
    s = s.replace(old, new)


rep('''#define TPB2_SPEC 0   // 1: the production configuration fixed at compile time (TP4, top-6, fused gate + fused shared expert, no in-kernel dsv3
                      // router; the host checks): its runtime branches fold away like TPB2_MMAX's''',
    '''#define TPB2_SPEC 0   // 1: the production configuration fixed at compile time (TP4, top-6, fused gate + fused shared expert, no in-kernel dsv3
                      // router; the host checks): its runtime branches fold away like TPB2_MMAX's.  (suite_b200) 2: routed-only (TP4,
                      // top-6, ids / weights given: no gate, no shared expert)''')
rep('__device__ __forceinline__ int pSHW(const Params& P) { return TPB2_SPEC ? 1 : P.shw; }',
    '__device__ __forceinline__ int pSHW(const Params& P) { return TPB2_SPEC == 2 ? 0 : TPB2_SPEC ? 1 : P.shw; }')
rep('__device__ __forceinline__ bool pGL(const Params& P) { return TPB2_SPEC ? true : P.glog != nullptr; }',
    '__device__ __forceinline__ bool pGL(const Params& P) { return TPB2_SPEC == 2 ? false : TPB2_SPEC ? true : P.glog != nullptr; }')
rep('''    TORCH_CHECK(!TPB2_SPEC || (P.ndev == NDEV && P.topk == 6 && P.shw && P.glog && !P.shared && (TPB2_SPEC_RF || !P.rF)), "TPB2_SPEC: TP4, top-6, fused gate + shared expert");''',
    '''    TORCH_CHECK(TPB2_SPEC != 1 || (P.ndev == NDEV && P.topk == 6 && P.shw && P.glog && !P.shared && (TPB2_SPEC_RF || !P.rF)), "TPB2_SPEC: TP4, top-6, fused gate + shared expert");
    TORCH_CHECK(TPB2_SPEC != 2 || (P.ndev == NDEV && P.topk == 6 && !P.shw && !P.glog && !P.shared && !P.rF && P.nparts == 1), "TPB2_SPEC=2: TP4, top-6, routed only (ids / weights given)");''')
P.write_text(s)
print('patched', P)
