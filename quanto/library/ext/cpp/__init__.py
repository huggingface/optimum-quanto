import os

import torch
from torch.library import impl
from torch.utils.cpp_extension import load


__all__ = []


_ext = None


def ext():
    """Helper to load the CPU ext only when it is required"""
    global _ext
    if _ext is None:
        module_path = os.path.dirname(__file__)
        _ext = load(
            name="quanto_cpp",
            sources=[
                f"{module_path}/unpack.cpp",
                f"{module_path}/pybind_module.cpp",
            ],
            extra_cflags=["-O3"],
        )
    return _ext


@impl("quanto_ext::unpack", ["CPU", "CUDA"])
def unpack_cpp(t: torch.Tensor, bits: int):
    return ext().unpack(t, bits)
