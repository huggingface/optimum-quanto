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

import ast
from typing import Optional

import torch
from torch.autograd import Function

from .qtensor import QTensor, qfallback
from .qtype import qtype, qtypes


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
    @staticmethod
    def create(
        qtype,
        axis,
        size,
        stride,
        data,
        scale,
        activation_qtype: Optional[qtype] = None,
        tensor_type: Optional[str] = None,
        requires_grad=False,
    ):
        """Factory method to create a QBytesTensor

        This selects the most appropriate QBytesTensor based on the configuration.

        Args:
            axis (`int`):
                The axis that is preserved by quantization (usually zero for linear weights).
            size ():
                The Tensor size.
            stride():
                The Tensor stride.
            data (`torch.Tensor`):
                The tensor data, either as a raw uint8 torch.Tensor or as a PackedTensor.
            scale (`torch.Tensor`):
                The floating point scale expressed as a torch.Tensor.
            activation_qtype (`qtype`):
            tensor_type (`Optional[str]`):
            requires_grad (`bool`):
                If the Tensor must be receive a gradient or not.

        Returns:
            a `QBytesTensor` (can be a subclass).
        """
        from .marlin import MarlinF8QBytesTensor

        init_cls = QBytesTensor
        if (
            tensor_type == "weight"
            and qtype == qtypes["qfloat8_e4m3fn"]
            and activation_qtype is None
            and scale.dtype in [torch.float16, torch.bfloat16]
            and len(size) == 2
            and data.device.type == "cuda"
            and axis is None
            and torch.cuda.get_device_capability(data.device)[0] >= 8
        ):
            if data.dtype == torch.int32 or (data.shape[0] % 64 == 0 and data.shape[1] % 16 == 0):
                init_cls = MarlinF8QBytesTensor

        return init_cls(qtype, axis, size, stride, data, scale, requires_grad)

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, requires_grad=False):
        assert data.device == scale.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, size, stride, data, scale, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = data
        self._scale = scale

    def __repr__(self):
        return f"{self.__class__}({self._data}, scale={self._scale}, dtype={self.dtype})"

    def dequantize(self):
        """Differentiable dequantization function"""
        return QBytesDequantizer.apply(self)
