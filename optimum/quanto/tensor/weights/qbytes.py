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

from ..function import QuantizedLinearFunction
from ..qbytes import QBytesTensor
from ..qtensor import qfallback
from ..qtype import qtype, qtypes


__all__ = ["WeightQBytesTensor"]


class WeightQBytesQuantizer(Function):

    @staticmethod
    def forward(
        ctx, base: torch.Tensor, qtype: qtype, axis: int, scale: torch.Tensor, activation_qtype: Optional[qtype]
    ) -> torch.Tensor:
        if qtype.bits != 8:
            raise ValueError("QBytesTensor can only be of 8-bit qtype")
        data = torch.ops.quanto.quantize_symmetric(base, dtype=qtype.dtype, axis=axis, scale=scale)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.

        return WeightQBytesTensor.create(
            qtype,
            axis,
            size=base.size(),
            stride=base.stride(),
            data=data,
            scale=scale,
            activation_qtype=activation_qtype,
        )

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None, None, None


class WeightQBytesLinearFunction(QuantizedLinearFunction):
    @staticmethod
    def forward(ctx, input, other, bias=None):
        ctx.save_for_backward(input, other)
        if isinstance(input, QBytesTensor):
            output = torch.ops.quanto.qbytes_mm(input._data, other._data, input._scale * other._scale)
        else:
            output = torch.ops.quanto.qbytes_mm(input, other._data, other._scale)
        if bias is not None:
            output = output + bias
        return output


class WeightQBytesTensor(QBytesTensor):
    @staticmethod
    def create(
        qtype,
        axis,
        size,
        stride,
        data,
        scale,
        activation_qtype: Optional[qtype] = None,
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
            activation_qtype (`qtype`, defaults to `None`):
                The qtype used for the activations. If one needs to use a different tensor subclass e.g. for weights depending on the activations qtype, this argument must be specified accordingly when calling `QBytesTensor.create`.
            requires_grad (`bool`):
                If the Tensor must be receive a gradient or not.

        Returns:
            a `QBytesTensor` (can be a subclass).
        """
        from .marlin import MarlinF8QBytesTensor

        init_cls = WeightQBytesTensor
        if (
            qtype == qtypes["qfloat8_e4m3fn"]
            and activation_qtype is None
            and scale.dtype in [torch.float16, torch.bfloat16]
            and len(size) == 2
            and data.device.type == "cuda"
            and axis == 0
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

    @classmethod
    def quantize(
        cls, base: torch.Tensor, qtype: qtype, axis: int, scale: torch.Tensor, activation_qtype: Optional[qtype] = None
    ) -> torch.Tensor:
        return WeightQBytesQuantizer.apply(base, qtype, axis, scale, activation_qtype)

    @staticmethod
    def load_from_state_dict(state_dict, prefix, qtype, axis, size, stride):
        inner_tensors_dict = {}
        for name in ["_data", "_scale"]:
            inner_tensors_dict[name] = state_dict.pop(prefix + name)
        meta = {
            "qtype": qtype.name,
            "axis": str(axis),
            "size": str(list(size)),
            "stride": str(list(stride)),
        }
        return WeightQBytesTensor.__tensor_unflatten__(inner_tensors_dict, meta, None, None)

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
        return WeightQBytesTensor(qtype, axis, size, stride, data, scale)

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
                return WeightQBytesLinearFunction.apply(input, other, bias)

            return qlinear(*args, **kwargs)
        # Defer to operations dispatcher
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Do not use directly op, but rather its overload
        op = op.overloadpacket
        if op is torch.ops.aten.detach:
            t = args[0]
            # Detach both data and scale
            out_data = op(t._data)
            out_scale = op(t._scale)
            return cls(t.qtype, t.axis, t.size(), t.stride(), out_data, out_scale)
        elif op in [torch.ops.aten._to_copy, torch.ops.aten.to]:
            t = args[0]
            dtype = kwargs.get("dtype", t.dtype)
            device = kwargs.get("device", t.device)
            # For data, ignore dtype and use the inner type instead
            out_data = op(t._data, dtype=t._data.dtype, device=device)
            out_scale = op(t._scale, dtype=dtype, device=device)
            return cls(t.qtype, t.axis, t.size(), t.stride(), out_data, out_scale)
        elif op is torch.ops.aten.t:
            t = args[0]
            out_data = op(t._data)
            out_scale = t._scale
            out_axis = t.axis
            # Manually reverse size and stride because we cannot trust the out_data shape
            dim0, dim1 = t.size()
            out_size = torch.Size([dim1, dim0])
            out_stride = t.stride()[::-1]
            if t.axis is not None:
                # We need to transpose also the scale
                out_scale = op(out_scale)
                out_axis = 0 if out_axis == -1 else -1
            return cls(t.qtype, out_axis, out_size, out_stride, out_data, out_scale)
        # No dispatch available: qfallback
        kwargs = kwargs or {}
        return qfallback(op, *args, **kwargs)
