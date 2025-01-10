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

from ...library import is_extension_available
from ..function import QuantizedLinearFunction
from ..qbytes import QBytesTensor
from ..qtensor import qfallback
from ..qtype import qtype, qtypes


__all__ = ["WeightQBytesTensor"]


class WeightQBytesQuantizer(Function):
    @staticmethod
    def forward(
        ctx, base: torch.Tensor, qtype: qtype, axis: int, scale: torch.Tensor, activation_qtype: qtype, optimized: bool
    ) -> torch.Tensor:
        if qtype.bits != 8:
            raise ValueError("QBytesTensor can only be of 8-bit qtype")
        data = torch.ops.quanto.quantize_symmetric(base, dtype=qtype.dtype, axis=axis, scale=scale)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.

        if optimized:
            return WeightQBytesTensor.create(
                qtype,
                axis,
                size=base.size(),
                stride=base.stride(),
                data=data,
                scale=scale,
                activation_qtype=activation_qtype,
            )
        return WeightQBytesTensor(
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
        return gO, None, None, None, None, None, None


class WeightQBytesLinearFunction(QuantizedLinearFunction):
    @staticmethod
    def forward(ctx, input, other, bias=None):
        ctx.save_for_backward(input, other)
        if isinstance(input, QBytesTensor):
            output = torch.ops.quanto.qbytes_mm(input._data, other._data, input._scale * other._scale)
        else:
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch.ops.quanto.qbytes_mm(input.reshape(-1, in_features), other._data, other._scale)
            output = output.reshape(output_shape)
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

        if (
            qtype == qtypes["qfloat8_e4m3fn"]
            and activation_qtype is None
            and scale.dtype in [torch.float16, torch.bfloat16]
            and len(size) == 2
            and (data.device.type == "cuda" and torch.version.cuda)
            and axis == 0
            and torch.cuda.get_device_capability(data.device)[0] >= 8
            and is_extension_available("quanto_cuda")
        ):
            out_features, in_features = size
            if (
                in_features >= 64
                and out_features >= 64
                and (
                    (in_features % 64 == 0 and out_features % 128 == 0)
                    or (in_features % 128 == 0 and out_features % 64 == 0)
                )
            ):
                return MarlinF8QBytesTensor(qtype, axis, size, stride, data, scale, requires_grad)

        return WeightQBytesTensor(qtype, axis, size, stride, data, scale, activation_qtype, requires_grad)

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, activation_qtype, requires_grad=False):
        assert data.device == scale.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, size, stride, data, scale, activation_qtype, requires_grad=False):
        super().__init__(qtype, axis, size, stride, data, scale, requires_grad=requires_grad)
        self.activation_qtype = activation_qtype

    @classmethod
    def quantize(
        cls,
        base: torch.Tensor,
        qtype: qtype,
        axis: int,
        scale: torch.Tensor,
        activation_qtype: Optional[qtype] = None,
        optimized: Optional[bool] = True,
    ) -> torch.Tensor:
        return WeightQBytesQuantizer.apply(base, qtype, axis, scale, activation_qtype, optimized)

    @staticmethod
    def load_from_state_dict(state_dict, prefix, qtype, axis, size, stride, activation_qtype, missing_keys):
        inner_tensors_dict = {}
        missing = False
        for name in ["_data", "_scale"]:
            if prefix + name not in state_dict:
                missing_keys.append(prefix + name)
                missing = True
            else:
                inner_tensors_dict[name] = state_dict.pop(prefix + name)

        if missing:  # could not deserialize because of missing keys
            return None

        meta = {
            "qtype": qtype.name,
            "axis": str(axis),
            "size": str(list(size)),
            "stride": str(list(stride)),
            "activation_qtype": "none" if activation_qtype is None else activation_qtype.name,
        }
        return WeightQBytesTensor.__tensor_unflatten__(inner_tensors_dict, meta, None, None)

    def optimize(self):
        """Allows to convert an existing WeightQBytesTensor to an optimized subclass

        This is used in particular after reloading a serialized WeightQBytesTensor (which is
        always saved using the kernel-agnostic packing).
        """
        if type(self) is not WeightQBytesTensor:
            return self
        # Call dedicated helper to select the best subclass for this device
        return WeightQBytesTensor.create(
            self.qtype,
            self.axis,
            self.size(),
            self.stride(),
            self._data,
            self._scale,
            self.activation_qtype,
            self.requires_grad,
        )

    def save_to_state_dict(self, destination, prefix, keep_vars):
        if type(self) is WeightQBytesTensor:
            super().save_to_state_dict(destination, prefix, keep_vars)
        else:
            # Convert back subclass before serializing
            self.weight_qbytes_tensor().save_to_state_dict(destination, prefix, keep_vars)

    def weight_qbytes_tensor(self):
        """Convert back a subclass to a WeightQBytesTensor

        This is required to make sure only standard packing is used when serializing.
        """
        raise NotImplementedError

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "activation_qtype": "none" if self.activation_qtype is None else self.activation_qtype.name,
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        assert len(meta) == 5
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        activation_qtype = None if meta["activation_qtype"] == "none" else qtypes[meta["activation_qtype"]]
        return WeightQBytesTensor(qtype, axis, size, stride, data, scale, activation_qtype)

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
        elif func is torch.equal:
            input, other = args
            return input.equal(other)
        # Defer to operations dispatcher
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Do not use directly op, but rather its overload
        op = op.overloadpacket
        if op is torch.ops.aten.detach:
            t = args[0]
            # Detach is required when copying and deserializing
            inner_tensor_names, meta = t.__tensor_flatten__()
            # Detach inner tensors
            detached_tensors = {}
            for inner_name in inner_tensor_names:
                detached_tensors[inner_name] = op(getattr(t, inner_name))
            return cls.__tensor_unflatten__(detached_tensors, meta, t.size(), t.stride())
        elif op in [torch.ops.aten._to_copy, torch.ops.aten.to]:
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype)
            device = kwargs.pop("device", t.device)
            if dtype != t.dtype:
                raise ValueError("The dtype of a weights Tensor cannot be changed")
            if type(t) is not WeightQBytesTensor and t.device.type != device.type:
                # Before moving to another device type, convert back to a WeightQBytesTensor
                t = t.weight_qbytes_tensor()
            out_data = op(t._data, device=device, **kwargs)
            out_scale = op(t._scale, device=device, **kwargs)
            return WeightQBytesTensor.create(
                t.qtype,
                t.axis,
                t.size(),
                t.stride(),
                out_data,
                out_scale,
                activation_qtype=t.activation_qtype,
                requires_grad=t.requires_grad,
            )
        elif op is torch.ops.aten.t and cls is WeightQBytesTensor:
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
            return WeightQBytesTensor(t.qtype, out_axis, out_size, out_stride, out_data, out_scale, t.activation_qtype)
        # No dispatch available: qfallback
        kwargs = kwargs or {}
        return qfallback(op, *args, **kwargs)
