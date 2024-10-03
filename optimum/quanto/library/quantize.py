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

from typing import Union

import torch

from ..tensor import dtype_info, group


torch.library.define(
    "quanto::quantize_symmetric", "(Tensor base, ScalarType dtype, int? axis, Tensor scale) -> Tensor"
)


@torch.library.impl("quanto::quantize_symmetric", "default")
def quantize_symmetric(
    base: torch.Tensor, dtype: torch.dtype, axis: Union[int, None], scale: torch.Tensor
) -> torch.Tensor:
    # Sanity checks
    if axis is None:
        if scale.ndim > 0:
            raise ValueError("Scale must be a scalar when quantizing per-tensor")
    else:
        if base.ndim == 1:
            raise ValueError("1D Tensors cannot be quantized per-axis")
        if axis == base.ndim - 1:
            # Align on the general convention to index the last dimension
            axis = -1
        if axis not in (0, -1):
            raise ValueError("Quantization is only supported along the first or last axis.")
        if base.shape[axis] == 1:
            raise ValueError(f"Cannot quantize Tensor of shape {base.shape} along axis {axis} of size 1")
        if torch.squeeze(scale).ndim > 1:
            raise ValueError("Quantizing along multiple axis is not supported")
        if scale.ndim != base.ndim:
            raise ValueError(
                "When quantizing per-axis, the scale must be broadcastable to the base (Tip: try to add missing dims of length zero)."
            )
    data = base / scale
    if not dtype.is_floating_point:
        data = torch.round(data)
    info = dtype_info(dtype)
    return torch.clamp(data, min=info.min, max=info.max).to(dtype)


torch.library.define(
    "quanto::quantize_affine",
    "(Tensor base, int bits, int axis, int? group_size, Tensor scale, Tensor shift) -> Tensor",
)


@torch.library.impl("quanto::quantize_affine", "default")
def quantize_affine(
    base: torch.Tensor, bits: int, axis: int, group_size: Union[int, None], scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    if axis not in (0, -1):
        raise ValueError("axis parameter must be 0 (first axis) or -1 (last axis)")
    if group_size is not None:
        base = group(base, axis=axis, group_size=group_size)
    if shift.dtype.is_floating_point:
        data = torch.round((base + shift) / scale)
    else:
        # Shift is an integer representing zero (i.e. zero-point)
        data = torch.round(base / scale) + shift

    return torch.clamp(data, min=0, max=2**bits - 1).to(torch.uint8)
