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

import torch
from torch.autograd import Function

from ..qbytes import QBytesTensor
from ..qtensor import qfallback
from ..qtype import qtype, qtypes


__all__ = ["ActivationQBytesTensor"]


class ActivationQBytesQuantizer(Function):
    @staticmethod
    def forward(ctx, base: torch.Tensor, qtype: qtype, scale: torch.Tensor) -> torch.Tensor:
        if qtype.bits != 8:
            raise ValueError("QBytesTensor can only be of 8-bit qtype")
        size = base.size()
        stride = base.stride()
        data = torch.ops.quanto.quantize_symmetric(base, dtype=qtype.dtype, axis=None, scale=scale)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.
        return ActivationQBytesTensor(qtype, size, stride, data, scale)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None, None, None


class ActivationQBytesTensor(QBytesTensor):
    @staticmethod
    def __new__(cls, qtype, size, stride, data, scale, requires_grad=False):
        assert data.device == scale.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, size, stride, data, scale, requires_grad=False):
        super().__init__(qtype, None, size, stride, data, scale, requires_grad)

    @classmethod
    def quantize(cls, base: torch.Tensor, qtype: qtype, scale: torch.Tensor) -> torch.Tensor:
        return ActivationQBytesQuantizer.apply(base, qtype, scale)

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale"]
        meta = {
            "qtype": self._qtype.name,
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        assert len(meta) == 3
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return ActivationQBytesTensor(qtype, size, stride, data, scale)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        from .qbytes_ops import get_qbytestensor_op_dispatch

        kwargs = kwargs or {}
        # Do not use directly op, but rather its overload
        op = op.overloadpacket
        qdispatch = get_qbytestensor_op_dispatch(op)
        if qdispatch is not None:
            return qdispatch(*args, **kwargs)
        # No dispatch available: qfallback
        return qfallback(op, *args, **kwargs)
