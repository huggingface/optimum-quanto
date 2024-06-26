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

import torch
from packaging import version


__all__ = []


torch.library.define("quanto::qbytes_mm", "(Tensor A, Tensor B, Tensor scales) -> Tensor")


def qbytes_mm(activations: torch.Tensor, weights: torch.Tensor, output_scales: torch.Tensor) -> torch.Tensor:
    mm_dtype = output_scales.dtype
    if activations.dtype == torch.int8 or weights.dtype == torch.int8:
        # If one of the terms is an int the matmul might overflow
        mm_dtype = torch.float32
    activations = activations.to(mm_dtype)
    weights = weights.to(mm_dtype)
    outputs = torch.matmul(activations, weights.t()) * output_scales.t()
    return outputs.to(output_scales.dtype)


def qbytes_int_mm(activations: torch.Tensor, weights: torch.Tensor, output_scales: torch.Tensor) -> torch.Tensor:
    in_features = activations.shape[-1]
    out_features = weights.shape[0]
    # torch._int_mm works on transposed weights, i.e (in_features, out_features)
    weights = weights.t()
    if activations.ndim == 2:
        out_data = torch._int_mm(activations, weights)
    else:
        output_shape = activations.shape[:-1] + (out_features,)
        out_data = torch._int_mm(activations.view(-1, in_features), weights)
        out_data = out_data.view(output_shape)
    # We must evaluate the output as float32 because the multiplication
    # of the int32 data by the scales might overflow
    fp32_output = out_data.to(torch.float32) * output_scales.t()
    return fp32_output.to(output_scales.dtype)


@torch.library.impl("quanto::qbytes_mm", "default")
def qbytes_mm_impl_default(
    activations: torch.Tensor, weights: torch.Tensor, output_scales: torch.Tensor
) -> torch.Tensor:
    return qbytes_mm(activations, weights, output_scales)


@torch.library.impl("quanto::qbytes_mm", "CUDA")
def qbytes_mm_impl_cuda(activations: torch.Tensor, weights: torch.Tensor, output_scales: torch.Tensor) -> torch.Tensor:
    assert activations.ndim in (2, 3)
    in_features = activations.shape[-1]
    tokens = activations.shape[0] if activations.ndim == 2 else activations.shape[0] * activations.shape[1]
    out_features = weights.shape[0]
    if (
        activations.dtype == torch.int8
        and weights.dtype == torch.int8
        and tokens > 16
        and tokens % 8 == 0
        and in_features % 8 == 0
        and out_features % 8 == 0
    ):
        return qbytes_int_mm(activations, weights, output_scales)
    return qbytes_mm(activations, weights, output_scales)


@torch.library.impl("quanto::qbytes_mm", "CPU")
def qbytes_mm_impl_cpu(activations: torch.Tensor, weights: torch.Tensor, output_scales: torch.Tensor) -> torch.Tensor:
    if (
        version.parse(torch.__version__).release >= version.parse("2.4.0").release
        and activations.dtype == torch.int8
        and weights.dtype == torch.int8
    ):
        return qbytes_int_mm(activations, weights, output_scales)
    return qbytes_mm(activations, weights, output_scales)
