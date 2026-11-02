"""Production router GEMM (DSV4-Flash layer router [256][4096] bf16, torch.mm(x, w.t(), out_dtype=fp32) / dsv3_router_gemm) reproduced
bitwise in tpmoe_b2: per M, the K parts whose tcgen05 kind::f16 K16 chains, summed in order (fp32), equal cuBLAS's split-K result.
Measured on B200 with this container's cuBLAS (b2_router_re.py / b2_router_re2.py); M <= 4 is sglang's dsv3_router_gemm (FMA chains).
None = not probed (keep the external router)."""


def router_parts(m):
    if m <= 4:
        return 'dsv3'
    if m <= 8 or m in (16, 32):
        return (1024, 1024, 1024, 1024)
    if m <= 24 or m in (48, 64):
        return (1408, 1408, 1280)
    if m <= 47:
        return (2048, 2048)
    if m <= 63 or m == 72:
        return (4096,)
    if m in (80, 96, 112, 128):
        return (2048, 2048)
    return None
