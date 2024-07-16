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
from copy import copy

import torch
from torch.autograd import Function

from ...qtensor import qfallback
from ...qtype import qtypes
from ..group import group, ungroup
from ..qbits import QBitsTensor
from .packed import TinyGemmPackedTensor


__all__ = ["TinyGemmQBitsTensor"]


class TinyGemmQBitsDequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        # There is no custom dequantize kernel available, so we need to convert back to a QBitsTensor
        qbt = t.qbits_tensor()
        return qbt.dequantize()

    @staticmethod
    def backward(ctx, gO):
        return gO


class TinyGemmQBitsTensor(QBitsTensor):
    @staticmethod
    def __new__(cls, qtype, axis, group_size, size, stride, data, scale_shift, requires_grad=False):
        if isinstance(scale_shift, torch.Tensor):
            dtype = scale_shift.dtype
            assert data.device == scale_shift.device
        else:
            assert isinstance(scale_shift, (tuple, list))
            scale, shift = scale_shift
            dtype = scale.dtype
            assert shift.dtype == dtype
            assert data.device == scale.device
            assert data.device == shift.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, group_size, size, stride, data, scale_shift, requires_grad=False):
        assert axis == 0
        if not isinstance(data, TinyGemmPackedTensor):
            assert type(data) is torch.Tensor
            assert isinstance(scale_shift, (tuple, list))
            # Format data, scale and shift for tinygemm
            ungrouped = ungroup(data, axis=0, orig_shape=size)
            self._data = TinyGemmPackedTensor.pack(ungrouped)
            out_features, in_features = size
            scale, shift = scale_shift
            scale = scale.reshape(out_features, in_features // group_size, 1)
            shift = shift.reshape(out_features, in_features // group_size, 1)
            if not shift.dtype.is_floating_point:
                # Integer shift must be scaled
                shift = scale * shift
            # The tinygemm kernel actually uses the mid-point of the quantization range as shift
            min_range = -shift
            half_qrange = 2 ** (qtype.bits - 1) * scale
            # This operation is lossy for bfloat16, and the actual value of shift will be lost
            shift = min_range + half_qrange
            # Scale and shift are actually stored in the same tensor
            self._scale_shift = torch.cat([scale, shift], 2).transpose(0, 1).contiguous()
        else:
            self._data = data
            self._scale_shift = scale_shift
        self._qtype = qtype
        self._axis = axis
        self._group_size = group_size

    def dequantize(self):
        return TinyGemmQBitsDequantizer.apply(self)

    def qbits_tensor(self):
        """Convert back to a QBitsTensor

        This is required to make sure only standard packing is used when serializing.
        """
        data = group(self._data.unpack(), axis=self.axis, group_size=self._group_size)
        n_scales = self._scale_shift.numel() // 2
        scale = self._scale_shift[:, :, 0].t().reshape((n_scales, 1))
        shift = self._scale_shift[:, :, 1].t().reshape((n_scales, 1))
        half_qrange = 2 ** (self.qtype.bits - 1) * scale
        # This operation is lossy for bfloat16, and the actual value of shift will not be recovered
        shift = half_qrange - shift
        return QBitsTensor(self._qtype, self._axis, self._group_size, self.size(), self.stride(), data, scale, shift)

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale_shift"]
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
        assert len(inner_tensors) == 2
        assert len(meta) == 5
        data, scale_shift = inner_tensors["_data"], inner_tensors["_scale_shift"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        group_size = ast.literal_eval(meta["group_size"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return TinyGemmQBitsTensor(qtype, axis, group_size, size, stride, data, scale_shift)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Do not use directly op, but rather its overload
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            scale_shift = op(t._scale_shift)
            return TinyGemmQBitsTensor(t._qtype, t._axis, t._group_size, t.size(), t.stride(), data, scale_shift)
        elif op.overloadpacket in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.get("dtype", None)
            if dtype is not None and dtype != t.dtype:
                raise ValueError("The dtype of a TinyGemmQBitsTensor cannot be changed")
            scale_shift = op(t._scale_shift, **kwargs)
            data_kwargs = copy(kwargs)
            data_kwargs["dtype"] = t._data.dtype
            data = op(t._data, **data_kwargs)
            return TinyGemmQBitsTensor(t._qtype, t._axis, t._group_size, t.size(), t.stride(), data, scale_shift)
        # No dispatch available: qfallback
        return qfallback(op, *args, **kwargs)
