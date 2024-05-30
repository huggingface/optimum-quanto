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

from ...qtype import qtypes
from ..group import group, ungroup
from ..qbits import QBitsTensor
from .packed import AWQPackedTensor, AWQPacking


__all__ = ["AWQBitsTensor"]


class AWQBitsDequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        unpacked = t._data.unpack()
        scale = t._scale
        zeropoint = t._zeropoint
        unpacked = group(unpacked, axis=0, group_size=t._group_size)
        n_scales = scale.numel()
        scale = scale.t().reshape((n_scales, 1))
        zeropoint = zeropoint.t().reshape((n_scales, 1))
        # Zeropoint are already scaled and negated
        dqt = scale * unpacked + zeropoint
        return ungroup(dqt, axis=t.axis, orig_shape=t.shape)

    @staticmethod
    def backward(ctx, gO):
        return gO


class AWQBitsTensor(QBitsTensor):
    @staticmethod
    def __new__(cls, qtype, axis, group_size, size, stride, data, scale, zeropoint, requires_grad=False):
        assert data.device.type == "cuda"
        assert data.device == scale.device
        assert data.device == zeropoint.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, group_size, size, stride, data, scale, zeropoint, requires_grad=False):
        assert axis == 0
        if not isinstance(data, AWQPackedTensor):
            assert type(data) == torch.Tensor
            # Format data, scale and zeropoint for optimized CUDA gemm
            ungrouped = ungroup(data, axis=0, orig_shape=size)
            data = AWQPackedTensor.pack(ungrouped, packing=AWQPacking.V2)
            out_features, in_features = size
            scale = scale.reshape(out_features, in_features // group_size).t().contiguous()
            zeropoint = zeropoint.reshape(out_features, in_features // group_size).t()
            # Zero-point are actually scaled to float16 and negated
            zeropoint = (-zeropoint * scale).contiguous()
        super().__init__(qtype, axis, group_size, size, stride, data, scale, zeropoint)

    def dequantize(self):
        return AWQBitsDequantizer.apply(self)

    def qbits_tensor(self):
        """Convert back to a QBitsTensor

        This is required to make sure only standard packing is used when serializing.
        """
        data = self._data.unpack()
        n_scales = self._scale.numel()
        scale = self._scale.t().reshape((n_scales, 1))
        zeropoint = self._zeropoint.t().reshape((n_scales, 1))
        return QBitsTensor(
            self._qtype, self._axis, self._group_size, self.size(), self.stride(), data, scale, zeropoint
        )

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_zeropoint"]
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
        data, scale, zeropoint = inner_tensors["_data"], inner_tensors["_scale"], inner_tensors["_zeropoint"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        group_size = ast.literal_eval(meta["group_size"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return AWQBitsTensor(qtype, axis, group_size, size, stride, data, scale, zeropoint)
