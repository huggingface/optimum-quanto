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

from ..qbits import QBitsTensor, group
from ..qtype import qint2, qint4, qtype


__all__ = ["AffineQuantizer"]


class AffineQuantizer(Function):
    """A standard affine quantizer."""

    @staticmethod
    def forward(
        ctx, base: torch.Tensor, qtype: qtype, axis: int, group_size: int, scale: torch.Tensor, zeropoint: torch.Tensor
    ):
        if qtype not in (qint2, qint4):
            raise ValueError("QBitsTensor can only be of qint2 or qint4 qtype")
        if axis not in (0, -1):
            raise ValueError("QBitsTensor axis parameter must be 0 (first axis) or -1 (last axis)")
        size = base.size()
        stride = base.stride()
        if group_size is not None:
            base = group(base, axis=axis, group_size=group_size)
        bits = qtype.bits
        data = torch.clamp(torch.round(base / scale) + zeropoint, min=0, max=2**bits - 1).to(torch.uint8)

        return QBitsTensor.create(qtype, axis, group_size, size, stride, data, scale, zeropoint)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None, None, None
