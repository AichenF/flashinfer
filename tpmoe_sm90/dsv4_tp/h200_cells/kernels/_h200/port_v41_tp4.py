"""port_v41_tp4.py: tp8v41.cu (H20-TP8 agent's port_v41.py output + its later patches, read-only copy) -> tpv41.cu: DSV4.1-Flash at
TP4 as well as TP8.  TP4: per-rank intermediate 576 = 4 items of 128 + a 64-column tail item (q = 4: gate rows 512..575 | up rows
512..575, one full 64-row Humming block per stack; block 1 of each stack out of bounds -> zero-filled), NKT2 5 (FC2 k-tile 4: slices
16, 17 + 2 zero-filled; TP2_FC2_T1S 2 -> KPipeT2: the tail k-tile as 2 pipelined k32 slices, kpipet2.cuh from gen_kpipe2.py),
reduce-scatter owner (rtile + t) % 4, slot rtile / 4 (5 slots); T4 split tail only at TP8.  TP8 builds: same code as tp8v41.cu.
usage: python3 port_v41_tp4.py SRC DST"""
import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text()
n = 0


def rep(old, new, count=1):
    global s, n
    c = s.count(old)
    assert c == count, f'expected {count}, found {c}: {old[:200]}'
    s = s.replace(old, new)
    n += 1


rep('static_assert(TP2_MODEL == 0 || NDEV == 8, "v41: TP8 only");\n', '')
rep('#define TP2_TAIL4 (TP2_MODEL == 1)   // v41: N8 tail items',
    '#define TP2_TAIL4 (TP2_MODEL == 1 && TP2_NDEV == 8)   // v41 TP8: N8 tail items')
rep('    if constexpr (TP2_MODEL != 0) return ((size_t)(src * MAXM + t) * NSLOT + (rtile >> 3)) * NMSG + m;',
    '    if constexpr (TP2_MODEL != 0) return ((size_t)(src * MAXM + t) * NSLOT + (rtile / NDEV)) * NMSG + m;   // (slot = rtile / NDEV)')
rep('#include "kpipe1.cuh"\n', '#include "kpipe1.cuh"\n#include "kpipet2.cuh"\n')
rep('''template <int N>
__device__ __forceinline__ void kfinish(uint64_t* empty, int sprev, float (&C)[N / 2], float (&D)[N / 2], int lane) {''',
    '''// FC2 tail k-tile of the TP4 split (TP2_FC2_T1S 2, NKT2 5): KPipeT2 = KPipe<N, false> cut after its second k32 slice
template <int N>
__device__ __forceinline__ void kstep_p2(uint8_t* smem, uint64_t* full, uint64_t* empty, int& s, unsigned& ph, int& sprev,
                                         float (&C)[N / 2], float (&D)[N / 2], uint64_t desc, int frag_off, int scl_off, int lane) {
    static_assert(KPS == 1, "kstep_p2: one k-tile per ring stage");
    mbar_wait(&full[s], ph);
    if (TP2_GATHER_FENCE) fence_proxy_async_smem();
    const uint8_t* st = smem + s * STAGE;
    kpipe::KPipeT2<N>::run(C, D, smem_u32(st + frag_off), smem_u32(st + scl_off), desc, smem_u32(&empty[sprev]), lane == 0 ? 1u : 0u);
    sprev = s;
    if (++s == NST) { s = 0; ph ^= 1; }
}
template <int N>
__device__ __forceinline__ void kfinish(uint64_t* empty, int sprev, float (&C)[N / 2], float (&D)[N / 2], int lane) {''')
rep('''    if constexpr (NKT2 >= 4)
        kstep<N, false>(smem, full, empty, s, ph, sprev, A1, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
    }
    kfinish<N>(empty, sprev, C, D, L.lane);''', '''    if constexpr (NKT2 >= 4)
        kstep<N, false>(smem, full, empty, s, ph, sprev, A1, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
    if constexpr (NKT2 == 5 && TP2_FC2_T1S == 2)   // (v41 TP4: k-tile 4 = slices 16, 17 + 2 zero-filled)
        kstep_p2<N>(smem, full, empty, s, ph, sprev, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane);
    else if constexpr (NKT2 >= 5)
        kstep<N, false>(smem, full, empty, s, ph, sprev, A0, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
    static_assert(NKT2 <= 5, "fc2_unit k-steps");
    }
    kfinish<N>(empty, sprev, C, D, L.lane);''')
Path(sys.argv[2]).write_text('// tpv41.cu = port_v41_tp4.py(tp8v41.cu): DSV4.1-Flash TP4 + TP8 (see port_v41_tp4.py)\n' + s)
print(f'{n} patches -> {sys.argv[2]}')
