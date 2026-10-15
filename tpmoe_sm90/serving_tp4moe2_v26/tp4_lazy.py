"""sitecustomize entry (FAC_FMOE_MODE=tp4moe2): install the hook only in processes that import the DeepSeek MoE module.

Importing sglang model code from sitecustomize would run in every Python process of the container (frontend, tokenizer,
detokenizer) and initialize CUDA there.  Instead, wait for sglang.srt.models.deepseek_v2 to finish executing (the TP
workers import it while resolving the model class, before weights load) and install install_tp4moe2 right after.
"""
import importlib.abc
import sys

TARGET = 'sglang.srt.models.deepseek_v2'


class _PostImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            import importlib.util
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module

        def exec_module(module):
            original(module)
            sys.meta_path.remove(self)
            import install_tp4moe2   # noqa: F401  (patches DeepseekV2MoE and the Humming weight post-processing)

        spec.loader.exec_module = exec_module
        return spec


if TARGET in sys.modules:
    import install_tp4moe2   # noqa: F401
else:
    sys.meta_path.insert(0, _PostImport())
