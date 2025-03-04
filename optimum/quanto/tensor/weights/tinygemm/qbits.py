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

from ...function import QuantizedLinearFunction
from ...grouped import group, ungroup
from ...qtype import qtypes
from ..qbits import WeightQBitsTensor
from .packed import TinyGemmPackedTensor


__all__ = ["TinyGemmWeightQBitsTensor"]


class TinyGemmQBitsDequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        # There is no custom dequantize kernel available, so we need to convert back to a QBitsTensor
        qbt = t.weight_qbits_tensor()
        return qbt.dequantize()

    @staticmethod
    def backward(ctx, gO):
        return gO


class TinyGemmQBitsLinearFunction(QuantizedLinearFunction):
    @staticmethod
    def forward(ctx, input, other, bias):
        ctx.save_for_backward(input, other)
        if type(input) is not torch.Tensor:
            input = input.dequantize()
        in_features = input.shape[-1]
        out_features = other.shape[0]
        output_shape = input.shape[:-1] + (out_features,)
        if input.device.type == "cpu":
            output = torch._weight_int4pack_mm_for_cpu(
                input.reshape(-1, in_features), other._data._data, other._group_size, other._scale_shift
            )
        else:
            output = torch._weight_int4pack_mm(
                input.reshape(-1, in_features), other._data._data, other._group_size, other._scale_shift
            )
        output = output.reshape(output_shape)
        if bias is not None:
            output = output + bias
        return output


class TinyGemmWeightQBitsTensor(WeightQBitsTensor):
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

    def weight_qbits_tensor(self):
        """Convert back to a WeightQBitsTensor

        This is required to make sure only standard packing is used when serializing.
        """
        data = group(self._data.unpack(), axis=self.axis, group_size=self._group_size)
        n_scales = self._scale_shift.numel() // 2
        scale = self._scale_shift[:, :, 0].t().reshape((n_scales, 1))
        shift = self._scale_shift[:, :, 1].t().reshape((n_scales, 1))
        half_qrange = 2 ** (self.qtype.bits - 1) * scale
        # This operation is lossy for bfloat16, and the actual value of shift will not be recovered
        shift = half_qrange - shift
        return WeightQBitsTensor(
            self._qtype, self._axis, self._group_size, self.size(), self.stride(), data, scale, shift
        )

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
        return TinyGemmWeightQBitsTensor(qtype, axis, group_size, size, stride, data, scale_shift)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Dispatch torch functions applied on this subtensor

        This method is called whenever a torch function (such as `torch.nn.functional.linear`)
        is called with at least one parameter coresponding to this subtensor:

        - if a quantized implementation exists for the selected function, it is called,
        - otherwise, the original implementation is called, deactivating further functional dispatch.

        During the execution of the standard torch function, a second-level of dispatch will
        happen, but this time directly on individual torch Tensor operations (mainly ATEN).
        """
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:

            def qlinear(input, other, bias=None):
                return TinyGemmQBitsLinearFunction.apply(input, other, bias)

            return qlinear(*args, **kwargs)
        # Defer to operations dispatcher
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
