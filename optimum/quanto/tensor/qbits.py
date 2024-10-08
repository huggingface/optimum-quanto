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

from .grouped import ungroup
from .packed import PackedTensor
from .qtensor import QTensor


__all__ = ["QBitsTensor"]


class QBitsDequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        if isinstance(t._data, PackedTensor):
            data = t._data.unpack()
        else:
            data = t._data
        shift = t._shift
        if not shift.dtype.is_floating_point:
            # Remove shift before multiplying by the scale
            data = data.to(torch.int8) - shift.to(torch.int8)
        if t.qtype.is_floating_point:
            # Upcast explicitly to the scale dtype
            dqt = t._scale * data.to(t._scale.dtype)
        else:
            dqt = t._scale * data
        if shift.dtype.is_floating_point:
            # Remove scaled shift
            dqt -= shift
        if t.axis is None:
            return dqt
        # Restore the original shape (if needed)
        return ungroup(dqt, axis=t.axis, orig_shape=t.shape)

    @staticmethod
    def backward(ctx, gO):
        return gO


class QBitsTensor(QTensor):
    def __init__(self, qtype, axis, group_size, size, stride, data, scale, shift, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = data
        self._scale = scale
        self._shift = shift
        self._group_size = group_size

    def __repr__(self):
        return f"{type(self).__name__}({self._data}, scale={self._scale}, shift={self._shift}, dtype={self.dtype})"

    def dequantize(self):
        return QBitsDequantizer.apply(self)
