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

from .qtensor import QTensor, qfallback
from .qtype import qtypes


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
        return f"QBytesTensor({self._data}, scale={self._scale}, dtype={self.dtype})"

    def dequantize(self):
        """Differentiable dequantization function"""
        return QBytesDequantizer.apply(self)

    @staticmethod
    def load_from_state_dict(state_dict, prefix):
        inner_tensors_dict = {}
        for name in ["_data", "_scale"]:
            inner_tensors_dict[name] = state_dict.pop(prefix + name)
        meta = [name.replace(prefix, "") for name in state_dict.keys() if name.startswith(prefix)]
        meta_dict = {}
        for name in meta:
            meta_dict[name] = state_dict.pop(prefix + name)
        return QBytesTensor.__tensor_unflatten__(inner_tensors_dict, meta_dict, None, None)

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        assert len(meta) == 4
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return QBytesTensor(qtype, axis, size, stride, data, scale)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        from .qbytes_ops import get_qbytestensor_op_dispatch

        # Do not use directly op, but rather its overload
        op = op.overloadpacket
        # Look for a dispatched op accepting QBytesTensor inputs
        qdispatch = get_qbytestensor_op_dispatch(op)
        if qdispatch is not None:
            return qdispatch(*args, **kwargs)
        # No dispatch available: qfallback
        return qfallback(op, *args, **kwargs)
