import os

import torch
from torch.utils.cpp_extension import load


__all__ = []


_ext = None


def ext():
    """Helper to load the CUDA ext only when it is required"""
    global _ext
    if _ext is None:
        module_path = os.path.dirname(__file__)
        _ext = load(name="quanto_cuda", sources=[f"{module_path}/unpack.cu", f"{module_path}/pybind_module.cpp",],)
    return _ext


@torch.library.impl("quanto_ext::unpack", ["CUDA"])
def unpack_cuda(t: torch.Tensor, bits: int):
    return ext().unpack(t, bits)
