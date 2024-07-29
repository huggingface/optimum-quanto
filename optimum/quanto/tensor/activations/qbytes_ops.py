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

import numbers
from functools import partial
from typing import Callable, List

import torch

from ..core import dtype_info
from ..qtensor import QTensor, qfallback
from ..qtype import qint8
from .qbytes import ActivationQBytesTensor
from .quantization import quantize_activation


__all__ = ["get_qbytestensor_op_dispatch", "register_qbytestensor_op"]


_QBYTESTENSOR_OP_TABLE = {}


def register_qbytestensor_op(aten_ops: List[Callable]):
    """
    Used for registering a new __torch_dispatch__ aten operation to QBytesTensor.

    The code to register a new operation looks like:

    @register_qbytestensor_op(list_of_ops)
    def foo(op, *args, **kwargs):
        <implementation>
    """

    def wrapper(op):
        for aten_op in aten_ops:
            _QBYTESTENSOR_OP_TABLE[aten_op] = partial(op, aten_op)

    return wrapper


def get_qbytestensor_op_dispatch(aten_op):
    return _QBYTESTENSOR_OP_TABLE.get(aten_op, None)


def is_scalar(t):
    return isinstance(t, numbers.Number) or type(t) is torch.Tensor and len(t.shape) == 0


@register_qbytestensor_op([torch.ops.aten._to_copy, torch.ops.aten.to])
def _to_copy(op, t, dtype=None, **kwargs):
    # For data, ignore dtype and use the inner type instead
    out_data = op(t._data, dtype=t._data.dtype, **kwargs)
    # Apply the new dtype on the scale only
    out_scale = op(t._scale, dtype=dtype, **kwargs)
    return ActivationQBytesTensor(t.qtype, t.size(), t.stride(), out_data, out_scale)


@register_qbytestensor_op([torch.ops.aten.detach])
def detach(op, t):
    # Detach both data and scale
    out_data = op(t._data)
    out_scale = op(t._scale)
    return ActivationQBytesTensor(t.qtype, t.size(), t.stride(), out_data, out_scale)


@register_qbytestensor_op([torch.ops.aten.cat])
def cat(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        # Only quantized tensors with identical scalar scales can be concatenated
        if (
            isinstance(t1, ActivationQBytesTensor)
            and isinstance(t2, ActivationQBytesTensor)
            and torch.equal(t1._scale, t2._scale)
            and t1.qtype == t2.qtype
        ):
            if t1.qtype.is_floating_point or t2.qtype.is_floating_point:
                # Cat is not supported for float8
                return qfallback(op, inputs, dim)
            out_data = op([t1._data, t2._data], dim)
            return ActivationQBytesTensor(t1.qtype, out_data.size(), out_data.stride(), out_data, t1._scale)
    return qfallback(op, inputs, dim)


@register_qbytestensor_op([torch.ops.aten.lt])
def lt(op, input, other):
    # Only quantized tensors with identical scales can be compared
    if (
        isinstance(input, ActivationQBytesTensor)
        and isinstance(other, ActivationQBytesTensor)
        and torch.equal(input._scale, other._scale)
    ):
        return op(input._data, other._data)
    return qfallback(op, input, other)


@register_qbytestensor_op([torch.ops.aten.clone])
def clone(op, t, memory_format=torch.preserve_format):
    # We need to restore the data original shape before cloning to get the correct strides
    data_shape = t._data.shape
    out_data = t._data.reshape(t.shape)
    out_data = op(t._data, memory_format=memory_format)
    out_stride = out_data.stride()
    out_data = out_data.reshape(data_shape)
    out_scale = op(t._scale, memory_format=memory_format)
    return ActivationQBytesTensor(t.qtype, t.size(), out_stride, out_data, out_scale)


@register_qbytestensor_op([torch.ops.aten.copy_])
def copy_(op, dest, src):
    assert dest.qtype == src.qtype
    dest._data = op(dest._data, src._data)
    dest._scale = op(dest._scale, src._scale)
    return dest


@register_qbytestensor_op([torch.ops.aten.div])
def div(op, input, other):
    if not is_scalar(other):
        return op(input.dequantize(), other)
    # We just divide the scale
    return ActivationQBytesTensor(input.qtype, input.size(), input.stride(), input._data, op(input._scale, other))


@register_qbytestensor_op([torch.ops.aten.neg])
def neg(op, input, *args, **kwargs):
    if input.qtype.is_floating_point:
        # Neg is not supported for float8
        return op(input.dequantize(), *args, **kwargs)
    out_data = op(input._data, *args, **kwargs)
    return ActivationQBytesTensor(input.qtype, input.size(), input.stride(), out_data, input._scale)


@register_qbytestensor_op(
    [
        torch.ops.aten.expand,
        torch.ops.aten.permute,
        torch.ops.aten.select,
        torch.ops.aten.slice,
        torch.ops.aten.unsqueeze,
    ]
)
def unary_type_agnostic_op(op, input, *args, **kwargs):
    if input.axis is not None:
        return op(input.dequantize(), *args, **kwargs)
    # When quantization is per-tensor, these operations can be transparently applied
    # without modifying the scale.
    out_data = op(input._data, *args, **kwargs)
    return ActivationQBytesTensor(input.qtype, out_data.size(), out_data.stride(), out_data, input._scale)


@register_qbytestensor_op([torch.ops.aten.is_same_size])
def is_same_size(op, input, other):
    a = input._data if isinstance(input, ActivationQBytesTensor) else input
    b = other._data if isinstance(other, ActivationQBytesTensor) else other
    return op(a, b)


def cannot_mm(t: QTensor):
    """True if the QTensor data cannot be passed to an mm op"""
    return t.axis is not None and t.size() != t._data.size()


@register_qbytestensor_op([torch.ops.aten.bmm])
def bmm(op, input, other):
    if not isinstance(input, ActivationQBytesTensor):
        return op(input, other.dequantize())
    if not isinstance(other, QTensor) or input.axis is not None:
        return op(input.dequantize(), other)
    if input.qtype != qint8 or other.qtype != qint8 or cannot_mm(other):
        return qfallback(op, input, other)
    # Cast data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = (input._scale * other._scale).to(torch.float32)
    return (out_data * out_scale).to(input._scale.dtype)


@register_qbytestensor_op([torch.ops.aten.mul])
def mul(op, input, other):
    # If one of the multiplicands is a scalar, just multiply the scale
    if is_scalar(input):
        return ActivationQBytesTensor(other.qtype, other.size(), other.stride(), other._data, input * other._scale)
    if is_scalar(other):
        return ActivationQBytesTensor(input.qtype, input.size(), input.stride(), input._data, other * input._scale)
    return qfallback(op, input, other)


@register_qbytestensor_op([torch.ops.aten.relu])
def relu(op, input):
    if input.qtype.is_floating_point:
        # Relu is not supported for float8 types
        return qfallback(op, input)
    out_data = op(input._data)
    return ActivationQBytesTensor(input.qtype, input.size(), input.stride(), out_data, input._scale)


@register_qbytestensor_op([torch.ops.aten._softmax])
def _softmax(op, input, dim, half_to_float):
    # Softmax must be performed in float
    float_data = op(input.dequantize(), dim, half_to_float)
    # Since softmax is normalized, we know the optimal scale

    out_scale = torch.tensor(1 / dtype_info(input.qtype.dtype).max, dtype=input._scale.dtype).to(input.device)
    return quantize_activation(float_data, qtype=input.qtype, scale=out_scale)


@register_qbytestensor_op([torch.ops.aten.stack])
def stack(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        # Only quantized tensors with identical scales can be stacked
        if (
            isinstance(t1, ActivationQBytesTensor)
            and isinstance(t2, ActivationQBytesTensor)
            and t1.axis is None
            and t2.axis is None
            and torch.equal(t1._scale, t2._scale)
            and t1.qtype == t2.qtype
        ):
            out_data = op([t1._data, t2._data], dim)
            return ActivationQBytesTensor(t1.qtype, out_data.size(), out_data.stride(), out_data, t1._scale)
    return qfallback(inputs, dim)


@register_qbytestensor_op([torch.ops.aten.split])
def split(op, input, *args, **kwargs):
    if input.axis is not None:
        return qfallback(op, input, *args, **kwargs)
    out_datas = op(input._data, *args, **kwargs)
    return [
        ActivationQBytesTensor(input.qtype, input.size(), input.stride(), out_data, input._scale)
        for out_data in out_datas
    ]


@register_qbytestensor_op([torch.ops.aten.transpose])
def transpose(op, input, *args):
    out_data = op(input._data, *args)
    out_size = out_data.size()
    out_stride = out_data.stride()
    out_scale = input._scale
    return ActivationQBytesTensor(input.qtype, out_size, out_stride, out_data, out_scale)


@register_qbytestensor_op([torch.ops.aten.t])
def transpose2d(op, input):
    out_data = op(input._data)
    out_scale = input._scale
    # Manually reverse size and stride because we cannot trust the out_data shape
    dim0, dim1 = input.size()
    out_size = torch.Size([dim1, dim0])
    out_stride = input.stride()[::-1]
    return ActivationQBytesTensor(input.qtype, out_size, out_stride, out_data, out_scale)


@register_qbytestensor_op([torch.ops.aten.view, torch.ops.aten._unsafe_view])
def view(op, input, *shape):
    if input.axis is None:
        # The view is transparent for QTensor with scalar scales
        out_data = op(input._data, *shape)
        return ActivationQBytesTensor(input.qtype, out_data.size(), out_data.stride(), out_data, input._scale)
    return qfallback(op, input, *shape)


@register_qbytestensor_op([torch.ops.aten.where])
def where(op, condition, input, other):
    if isinstance(condition, QTensor) or isinstance(other, QTensor):
        raise NotImplementedError
    float_data = op(condition, input.dequantize(), other)
    if input.axis is None:
        # We requantize with the input scale
        return quantize_activation(float_data, qtype=input.qtype, scale=input._scale)
    return float_data
