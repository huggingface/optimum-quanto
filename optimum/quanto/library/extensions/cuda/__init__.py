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

from ..extension import Extension, register_extension


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


extra_cflags = ["-g", "-O3"]
extra_cuda_cflags = [
    "--expt-extended-lambda",
    "--use_fast_math",
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
    "marlin/fp8_marlin.cu",
    "marlin/gptq_marlin_repack.cu",
    "marlin/marlin_cuda.cpp",
    "marlin/marlin_cuda_kernel.cu",
    "pybind_module.cpp",
]
ext = Extension(
    "quanto_cuda",
    root_dir=os.path.dirname(__file__),
    sources=sources,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
)
register_extension(ext)


@torch.library.impl("quanto::unpack", ["CUDA"])
def unpack_cuda(t: torch.Tensor, bits: int):
    return ext.lib.unpack(t, bits)


torch.library.define(
    "quanto::gemm_f16i4_awq",
    "(Tensor input,"
    " Tensor other,"
    " Tensor other_scale,"
    " Tensor other_shift,"
    " int rows,"
    " int out_cols,"
    " int in_cols,"
    " int bits,"
    " int group_size)"
    " -> Tensor",
)


@torch.library.impl("quanto::gemm_f16i4_awq", ["CUDA"])
def gemm_f16i4_awq(
    input: torch.Tensor,
    other: torch.Tensor,
    scales: torch.Tensor,
    shift: torch.Tensor,
    rows: int,
    out_cols: int,
    in_cols: int,
    bits: int,
    group_size: int,
):
    assert out_cols >= 128
    assert input.dtype == torch.float16
    assert input.numel() == rows * in_cols
    assert other.dtype == torch.int16
    assert scales.dtype == torch.float16
    assert scales.shape[-1] == out_cols
    assert shift.dtype == torch.float16
    assert shift.shape[-1] == out_cols
    assert bits == 4
    assert group_size == 128
    if rows < 8:
        return ext.lib.awq_v2_gemv_f16i4(input, other, scales, shift, rows, out_cols, in_cols, group_size)
    return ext.lib.awq_v2_gemm_f16i4(input, other, scales, shift)


torch.library.define(
    "quanto::gemm_f16f8_marlin",
    "(Tensor a,"
    "Tensor b_q_weight,"
    "Tensor b_scales,"
    "Tensor workspace,"
    "int num_bits,"
    "int size_m,"
    "int size_n,"
    "int size_k)"
    " -> Tensor",
)


@torch.library.impl("quanto::gemm_f16f8_marlin", ["CUDA"])
def fp8_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    assert b_scales.dtype == torch.float16 or b_scales.dtype == torch.bfloat16
    assert b_q_weight.dim() == 2
    assert b_q_weight.dtype == torch.int32
    return ext.lib.fp8_marlin_gemm(a, b_q_weight, b_scales, workspace, num_bits, size_m, size_n, size_k)


torch.library.define(
    "quanto::pack_fp8_marlin",
    "(Tensor b_q_weight, Tensor perm, int size_k, int size_n, int num_bits) -> Tensor",
)


@torch.library.impl("quanto::pack_fp8_marlin", ["CUDA"])
def gptq_marlin_repack(
    b_q_weight: torch.Tensor, perm: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    assert b_q_weight.dim() == 2
    assert b_q_weight.dtype == torch.int32
    return ext.lib.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)


torch.library.define(
    "quanto::gemm_f16i4_marlin",
    "(Tensor input, Tensor other, Tensor other_scale, Tensor other_shift, Tensor workspace) -> Tensor",
)


@torch.library.impl("quanto::gemm_f16i4_marlin", ["CUDA"])
def gemm_f16i4_marlin(
    input: torch.Tensor, other: torch.Tensor, scales: torch.Tensor, zeropoint: torch.Tensor, workspace: torch.Tensor
) -> torch.Tensor:
    assert input.dtype == torch.float16
    assert other.dtype == torch.int32
    assert scales.dtype == torch.float16
    assert zeropoint.dtype == torch.float16
    assert workspace.dtype == torch.int32
    output = torch.empty(
        input.shape[:-1] + (scales.shape[1],),
        dtype=input.dtype,
        device=input.device,
    )
    ext.lib.marlin_gemm_f16i4(
        input.reshape((-1, input.shape[-1])),
        other,
        output.reshape((-1, output.shape[-1])),
        scales,
        zeropoint,
        workspace,
        -1,
        -1,
        -1,
        16,
    )
    return output
