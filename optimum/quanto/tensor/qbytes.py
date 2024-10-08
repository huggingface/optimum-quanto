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

from torch.autograd import Function

from .qtensor import QTensor


__all__ = ["QBytesTensor"]


class QBytesDequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        if t.qtype.is_floating_point:
            # Upcast explicitly to the scale dtype
            dqt = t._scale * t._data.to(t._scale.dtype)
        else:
            dqt = t._scale * t._data
        return dqt

    @staticmethod
    def backward(ctx, gO):
        # For autograd, dequantization is a no-op
        return gO


class QBytesTensor(QTensor):
    def __init__(self, qtype, axis, size, stride, data, scale, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = data
        self._scale = scale

    def __repr__(self):
        return f"{self.__class__}({self._data}, scale={self._scale}, dtype={self.dtype})"

    def dequantize(self):
        """Differentiable dequantization function"""
        return QBytesDequantizer.apply(self)
