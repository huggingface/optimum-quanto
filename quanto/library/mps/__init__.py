import os

import torch
from torch.library import impl
from torch.utils.cpp_extension import load

from ..ops import quanto_ops


__all__ = []


module_path = os.path.dirname(__file__)
mps_lib = load(name="quanto_mps", sources=[f"{module_path}/unpack.mm"], extra_cflags=["-std=c++17"])


@impl(quanto_ops, "unpack", "MPS")
def unpack_mps(t: torch.Tensor, bits: int):
    return mps_lib.unpack(t, bits)
