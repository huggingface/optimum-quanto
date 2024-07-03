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

from ..extension import Extension


__all__ = []


def get_max_cuda_arch():
    """Select the maximum CUDA arch supported

    This is a combination of the CUDA and pytorch version and all detected devices capabilities.
    """
    capability_list = []
    supported_sm = [int(arch.split("_")[1]) for arch in torch.cuda.get_arch_list() if "sm_" in arch]
    if supported_sm:
        max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            # Capability of the device may be higher than what's supported by the user's
            # NVCC, causing compilation error. User's NVCC is expected to match the one
            # used to build pytorch, so we use the maximum supported capability of pytorch
            # to clamp the capability.
            capability = min(max_supported_sm, capability)
            if capability not in capability_list:
                capability_list.append(capability)
    max_capability = max(sorted(capability_list)) if len(capability_list) > 0 else (0, 0)
    return f"{max_capability[0]}{max_capability[1]}0"


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
# We need to know the minimum CUDA Arch to select only the relevant kernels
# but we cannot rely on __CUDA_ARCH__ as it is not set in host code (only on device code)
quanto_cuda_arch = get_max_cuda_arch()
extra_cuda_cflags += [f"-DQUANTO_CUDA_ARCH={quanto_cuda_arch}"]
module_path = os.path.dirname(__file__)
sources = [
    "unpack.cu",
    "awq/v2/gemm_cuda.cu",
    "awq/v2/gemv_cuda.cu",
    "pybind_module.cpp",
]
ext = Extension(
    "quanto_cuda",
    root_dir=os.path.dirname(__file__),
    sources=sources,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
)


@torch.library.impl("quanto_ext::unpack", ["CUDA"])
def unpack_cuda(t: torch.Tensor, bits: int):
    return ext.lib.unpack(t, bits)


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
        return ext.lib.awq_v2_gemv_f16i4(input, other, scales, zeropoint, rows, out_cols, in_cols, group_size)
    return ext.lib.awq_v2_gemm_f16i4(input, other, scales, zeropoint)
