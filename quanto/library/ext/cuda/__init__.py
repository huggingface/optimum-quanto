# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.cpp_extension import load


__all__ = []


_ext = None


def ext():
    """Helper to load the CUDA ext only when it is required"""
    global _ext
    if _ext is None:
        extra_cflags = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
        extra_cuda_cflags = [
            "-O3",
            "-std=c++17",
            "-DENABLE_BF16",  # TODO
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--threads=8",
        ]
        module_path = os.path.dirname(__file__)
        _ext = load(
            name="quanto_cuda",
            sources=[
                f"{module_path}/unpack.cu",
                f"{module_path}/awq/v2/gemm_cuda.cu",
                f"{module_path}/awq/v2/gemv_cuda.cu",
                f"{module_path}/pybind_module.cpp",
            ],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
        )
    return _ext


@torch.library.impl("quanto_ext::unpack", ["CUDA"])
def unpack_cuda(t: torch.Tensor, bits: int):
    return ext().unpack(t, bits)


@torch.library.impl("quanto_ext::gemm", ["CUDA"])
def gemm_cuda(
    input: torch.Tensor,
    other: torch.Tensor,
    scales: torch.Tensor,
    zeropoint: torch.Tensor,
    rows: int,
    out_cols: int,
    in_cols: int,
    bits: int,
    group_size: int,
):
    assert input.dtype == torch.float16
    assert input.numel() == rows * in_cols
    assert other.dtype == torch.int16
    assert scales.dtype == torch.float16
    assert scales.shape[-1] == out_cols
    assert zeropoint.dtype == torch.float16
    assert zeropoint.shape[-1] == out_cols
    assert bits == 4
    assert group_size == 128
    if rows < 8:
        return ext().awq_v2_gemv_f16i4(input, other, scales, zeropoint, rows, out_cols, in_cols, group_size)
    return ext().awq_v2_gemm_f16i4(input, other, scales, zeropoint)
