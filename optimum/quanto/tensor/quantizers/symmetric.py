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
from torch.autograd import Function

from ..core import dtype_info
from ..qbytes import QBytesTensor
from ..qtype import qtype


__all__ = ["SymmetricQuantizer"]


class SymmetricQuantizer(Function):
    """A standard symmetric quantizer."""

    @staticmethod
    def forward(ctx, base: torch.Tensor, qtype: qtype, axis: int, scale: torch.Tensor):
        size = base.size()
        stride = base.stride()
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
                raise ValueError("QBytesTensor can only be quantized along the first or last axis.")
            if base.shape[axis] == 1:
                raise ValueError(f"Cannot quantize Tensor of shape {base.shape} along axis {axis} of size 1")
            if torch.squeeze(scale).ndim > 1:
                raise ValueError("Quantizing along multiple axis is not supported")
            if scale.ndim != base.ndim:
                raise ValueError(
                    "When quantizing per-axis, the scale must be broadcastable to the base (Tip: try to add missing dims of length zero)."
                )
        data = base / scale
        if not qtype.is_floating_point:
            data = torch.round(data)
        info = dtype_info(qtype.dtype)
        data = torch.clamp(data, min=info.min, max=info.max).to(qtype.dtype)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.
        return QBytesTensor(qtype, axis, size, stride, data, scale)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None, None
