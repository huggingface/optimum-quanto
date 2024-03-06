import os

import torch
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
                f"{module_path}/mm.cpp",
                f"{module_path}/mm_4bit.cpp",
                f"{module_path}/mm_2bit.cpp",
                f"{module_path}/quantize.cpp",
                f"{module_path}/unpack.cpp",
                f"{module_path}/pybind_module.cpp",
            ],
            extra_cflags=["-O3"],
        )
    return _ext


@torch.library.impl("quanto_ext::dqmm", ["CPU", "CUDA", "MPS"])
def dqmm_cpp(input: torch.Tensor, other: torch.Tensor, other_scale: torch.Tensor):
    return ext().dqmm(input, other, other_scale)


@torch.library.impl("quanto_ext::quantize_symmetric", ["CPU"])
def quantize_symmetric_cpp(t: torch.Tensor, scale: torch.Tensor, dtype: torch.Tensor.dtype):
    return ext().quantize_symmetric(t, scale, dtype)


@torch.library.impl("quanto_ext::unpack", ["CPU", "CUDA"])
def unpack_cpp(t: torch.Tensor, bits: int):
    return ext().unpack(t, bits)

@torch.library.impl("quanto_ext::mm_4bit", ["CPU", "CUDA"])
def mm_4bit_cpp(input: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor):
    return ext().mm_4bit(input, weight, scales)

@torch.library.impl("quanto_ext::mm_2bit", ["CPU", "CUDA"])
def mm_2bit_cpp(input: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor):
    return ext().mm_2bit(input, weight, scales)