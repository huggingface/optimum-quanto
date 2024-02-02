import os

import torch
from torch.library import impl
from torch.utils.cpp_extension import load

from ..ops import quanto_ops


__all__ = []


_backend = None


def backend():
    """Helper to load the CPU backend only when it is required"""
    global _backend
    if _backend is None:
        module_path = os.path.dirname(__file__)
        _backend = load(
            name="quanto_cpu",
            sources=[
                f"{module_path}/unpack.cpp",
                f"{module_path}/pybind_module.cpp",
            ],
            extra_cflags=["-O3"],
        )
    return _backend


@impl(quanto_ops, "unpack", "CPU")
@impl(quanto_ops, "unpack", "CUDA")
def unpack_cpu(t: torch.Tensor, bits: int):
    return backend().unpack(t, bits)
