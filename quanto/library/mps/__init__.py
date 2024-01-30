import os

import torch
from torch.library import impl
from torch.utils.cpp_extension import load

from ..ops import quanto_ops


__all__ = []


_backend = None


def backend():
    """Helper to load the MPS backend only when it is required"""
    global _backend
    if _backend is None:
        module_path = os.path.dirname(__file__)
        _backend = load(name="quanto_mps", sources=[f"{module_path}/unpack.mm"], extra_cflags=["-std=c++17"])
    return _backend


@impl(quanto_ops, "unpack", "MPS")
def unpack_mps(t: torch.Tensor, bits: int):
    return backend().unpack(t, bits)
