import os

import torch
from torch.library import impl
from torch.utils.cpp_extension import load

from ..ops import quanto_ops


__all__ = []


module_path = os.path.dirname(__file__)
cpu_lib = load(name="quanto_cpu", sources=[f"{module_path}/unpack.cpp"], extra_cflags=["-O3"])


@impl(quanto_ops, "unpack", "CPU")
@impl(quanto_ops, "unpack", "CUDA")
def unpack_cpu(t: torch.Tensor, bits: int):
    return cpu_lib.unpack(t, bits)
