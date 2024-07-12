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

from ..qtensor import QTensor, qfallback
from ..qtype import qint4, qtypes
from .group import grouped_shape, ungroup
from .packed import PackedTensor


__all__ = ["QBitsTensor"]


class QBitsDequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        data = t._data.unpack()
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
    @staticmethod
    def create(qtype, axis, group_size, size, stride, data, scale, shift, requires_grad=False):
        """Factory method to create a QBitsTensor

        This selects the most appropriate QBitsTensor based on the configuration.

        Args:
            axis (`int`):
                The axis that is preserved by quantization (usually zero for linear weights).
            group_size (`int`):
                The group size that further splits the data elements for each index along the quantization axis.
            size ():
                The Tensor size.
            stride():
                The Tensor stride.
            data (`torch.Tensor`):
                The tensor data, either as a raw uint8 torch.Tensor or as a PackedTensor.
            scale (`torch.Tensor`):
                The floating point scale expressed as a torch.Tensor.
            shift (`torch.Tensor`):
                The shift expressed as a torch.Tensor. It can be either an integer representing zero
                (i.e. zero-point) or a float value.
            requires_grad (`bool`):
                If the Tensor must be receive a gradient or not.

        Returns:
            a `QBitsTensor` (can be a subclass).
        """
        from .awq import AWQBitsTensor
        from .tinygemm import TinyGemmQBitsTensor

        if (
            qtype == qint4
            and scale.dtype == torch.float16
            and axis == 0
            and group_size == 128
            and len(size) == 2
            and data.device.type == "cuda"
            and torch.cuda.get_device_capability(data.device)[0] >= 8
        ):
            if type(data) == PackedTensor:
                data = data.unpack()
            return AWQBitsTensor(qtype, axis, group_size, size, stride, data, scale, shift, requires_grad)
        if qtype == qint4 and scale.dtype == torch.bfloat16 and axis == 0 and group_size == 128 and len(size) == 2:
            if data.device.type == "cpu" or (
                data.device.type == "cuda" and torch.cuda.get_device_capability(data.device)[0] >= 8
            ):
                if type(data) == PackedTensor:
                    data = data.unpack()
                return TinyGemmQBitsTensor(qtype, axis, group_size, size, stride, data, (scale, shift), requires_grad)

        return QBitsTensor(qtype, axis, group_size, size, stride, data, scale, shift, requires_grad)

    @staticmethod
    def __new__(cls, qtype, axis, group_size, size, stride, data, scale, shift, requires_grad=False):
        assert data.device == scale.device
        assert data.device == shift.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, group_size, size, stride, data, scale, shift, requires_grad=False):
        super().__init__(qtype, axis)
        if type(data) == torch.Tensor:
            data = PackedTensor.pack(data, qtype.bits)
        self._data = data
        self._scale = scale
        self._shift = shift
        self._group_size = group_size

    def __repr__(self):
        return f"{type(self).__name__}({self._data}, scale={self._scale}, shift={self._shift}, dtype={self.dtype})"

    def dequantize(self):
        return QBitsDequantizer.apply(self)

    @staticmethod
    def load_from_state_dict(state_dict, prefix, qtype, axis, group_size, size, stride):
        data_size = size if group_size is None else grouped_shape(size, axis, group_size)
        # In row major, inner dimension (stride 1) is the last one
        data_stride = (data_size[1], 1)
        inner_tensors_dict = {
            "_data": PackedTensor.load_from_state_dict(
                state_dict, prefix + "_data.", qtype.bits, data_size, data_stride
            )
        }
        for name in ["_scale", "_shift"]:
            inner_tensors_dict[name] = state_dict.pop(prefix + name)
        meta = {
            "qtype": qtype.name,
            "axis": str(axis),
            "group_size": str(group_size),
            "size": str(list(size)),
            "stride": str(list(stride)),
        }
        return QBitsTensor.__tensor_unflatten__(inner_tensors_dict, meta, None, None)

    def optimize(self):
        """Allows to convert an existing QBitsTensor to an optimized subclass"""
        if type(self) != QBitsTensor:
            return self
        data = self._data.unpack()
        # Call dedicated helper to select the best subclass for this device
        return QBitsTensor.create(
            self.qtype,
            self.axis,
            self._group_size,
            self.size(),
            self.stride(),
            data,
            self._scale,
            self._shift,
            self.requires_grad,
        )

    def save_to_state_dict(self, destination, prefix, keep_vars):
        if type(self) == QBitsTensor:
            super().save_to_state_dict(destination, prefix, keep_vars)
        else:
            # Convert back subclass before serializing
            self.qbits_tensor().save_to_state_dict(destination, prefix, keep_vars)

    def qbits_tensor(self):
        """Convert back a subclass to a QBitsTensor

        This is required to make sure only standard packing is used when serializing.
        """
        raise NotImplementedError

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_shift"]
        # Since meta can be used for serialization, use only strings
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "group_size": str(self._group_size),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 3
        assert len(meta) == 5
        data, scale, shift = inner_tensors["_data"], inner_tensors["_scale"], inner_tensors["_shift"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        group_size = ast.literal_eval(meta["group_size"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return QBitsTensor(qtype, axis, group_size, size, stride, data, scale, shift)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        from .qbits_ops import get_qbitstensor_op_dispatch

        # Do not use directly op, but rather its overload
        op = op.overloadpacket
        # Look for a dispatched op accepting QBitsTensor inputs
        qdispatch = get_qbitstensor_op_dispatch(op)
        if qdispatch is not None:
            return qdispatch(*args, **kwargs)
        # No dispatch available: qfallback
        return qfallback(op, *args, **kwargs)
