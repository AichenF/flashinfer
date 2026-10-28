"""Load (or build) the two frozen serving extensions; shared by the hook and build.py so both make the identical load() calls.
sh: tp4moe2_sh.cu (fused shared expert, used for M <= 32); ns: tp4moe2_ns.cu built with -DTP2_SHARED=0 (no shared code: the routed
loops at full speed, used for larger M).  Both are built from tp4moe2_ns.cu (sh: TP2_SHARED=1, its default).  ndev 8 (TP8):
-DTP2_NDEV=8, under their own names and build directories."""
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent


def load_ext(kind, ndev=4):
    from torch.utils.cpp_extension import load
    assert ndev in (4, 8) and kind in ('ns', 'sh'), (kind, ndev)
    frozen = json.loads((OUT / 'sources.json').read_text())
    tag = '' if ndev == 4 else f'_tp{ndev}'
    build = OUT / ('build_' + kind + tag)
    build.mkdir(exist_ok=True)
    flags = (['-DTP2_SHARED=0'] if kind == 'ns' else []) + ([f'-DTP2_NDEV={ndev}'] if ndev != 4 else [])
    saved = os.environ.get('TORCH_CUDA_ARCH_LIST')
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'   # sm_90a only (wgmma / setmaxnreg)
    try:
        return load(name=frozen['extension_' + kind] + tag, sources=[str(OUT / 'tp4moe2_ns.cu')], build_directory=str(build),
                    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-I{OUT}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'] + flags,
                    verbose=False)
    finally:
        if saved is None:
            del os.environ['TORCH_CUDA_ARCH_LIST']
        else:
            os.environ['TORCH_CUDA_ARCH_LIST'] = saved
