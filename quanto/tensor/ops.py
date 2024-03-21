import numbers
from functools import partial
from typing import Callable, List

import torch

from .core import dtype_info
from .qtensor import QTensor, qfallback
from .qtype import qint8
from packaging import version

__all__ = ["get_qtensor_op_dispatch", "register_qtensor_op"]


_QTENSOR_OP_TABLE = {}


def register_qtensor_op(aten_ops: List[Callable]):
    """
    Used for registering a new __torch_dispatch__ aten operation to QTensor.

    The code to register a new operation looks like:

    @register_qtensor_op(list_of_ops)
    def foo(op, *args, **kwargs):
        <implementation>
    """

    def wrapper(op):
        for aten_op in aten_ops:
            _QTENSOR_OP_TABLE[aten_op] = partial(op, aten_op)

    return wrapper


def get_qtensor_op_dispatch(aten_op):
    return _QTENSOR_OP_TABLE.get(aten_op, None)


def is_scalar(t):
    return isinstance(t, numbers.Number) or type(t) == torch.Tensor and len(t.shape) == 0


@register_qtensor_op([torch.ops.aten._to_copy])
def _to_copy(op, t, dtype=None, **kwargs):
    # For data, ignore dtype and use the inner type instead
    out_data = op(t._data, dtype=t._data.dtype, **kwargs)
    # Apply the new dtype on the scale only
    out_scale = op(t._scale, dtype=dtype, **kwargs)
    return QTensor(t.qtype, t.axis, t.size(), t.stride(), out_data, out_scale)


@register_qtensor_op([torch.ops.aten.detach])
def detach(op, t):
    # Detach both data and scale
    out_data = op(t._data)
    out_scale = op(t._scale)
    return QTensor(t.qtype, t.axis, t.size(), t.stride(), out_data, out_scale)


@register_qtensor_op([torch.ops.aten.cat])
def cat(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        # Only quantized tensors with identical scalar scales can be concatenated
        if (
            isinstance(t1, QTensor)
            and isinstance(t2, QTensor)
            and t1.axis is None
            and t2.axis is None
            and torch.equal(t1._scale, t2._scale)
            and t1.qtype == t2.qtype
        ):
            if t1.qtype.is_floating_point or t2.qtype.is_floating_point:
                # Cat is not supported for float8
                return qfallback(op, inputs, dim)
            out_data = op([t1._data, t2._data], dim)
            return QTensor(t1.qtype, t1.axis, out_data.size(), out_data.stride(), out_data, t1._scale)
    return qfallback(op, inputs, dim)


@register_qtensor_op([torch.ops.aten.lt])
def lt(op, input, other):
    # Only quantized tensors with identical scales can be compared
    if isinstance(input, QTensor) and isinstance(other, QTensor) and torch.equal(input._scale, other._scale):
        return op(input._data, other._data)
    return qfallback(op, input, other)


@register_qtensor_op([torch.ops.aten.clone])
def clone(op, t, memory_format=torch.preserve_format):
    # We need to restore the data original shape before cloning to get the correct strides
    data_shape = t._data.shape
    out_data = t._data.reshape(t.shape)
    out_data = op(t._data, memory_format=memory_format)
    out_stride = out_data.stride()
    out_data = out_data.reshape(data_shape)
    out_scale = op(t._scale, memory_format=memory_format)
    return QTensor(t.qtype, t.axis, t.size(), out_stride, out_data, out_scale)


@register_qtensor_op([torch.ops.aten.copy_])
def copy_(op, dest, src):
    assert dest.qtype == src.qtype
    dest._data = op(dest._data, src._data)
    dest._scale = op(dest._scale, src._scale)
    return dest


@register_qtensor_op([torch.ops.aten.div])
def div(op, input, other):
    if not is_scalar(other):
        return op(input.dequantize(), other)
    # We just divide the scale
    return QTensor(input.qtype, input.axis, input.size(), input.stride(), input._data, op(input._scale, other))


@register_qtensor_op([torch.ops.aten.neg])
def neg(op, input, *args, **kwargs):
    if input.qtype.is_floating_point:
        # Neg is not supported for float8
        return op(input.dequantize(), *args, **kwargs)
    out_data = op(input._data, *args, **kwargs)
    return QTensor(input.qtype, input.axis, input.size(), input.stride(), out_data, input._scale)


@register_qtensor_op(
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
    return QTensor(input.qtype, input.axis, out_data.size(), out_data.stride(), out_data, input._scale)


@register_qtensor_op([torch.ops.aten.is_same_size])
def is_same_size(op, input, other):
    a = input._data if isinstance(input, QTensor) else input
    b = other._data if isinstance(other, QTensor) else other
    return op(a, b)


def cannot_mm(t: QTensor):
    """True if the QTensor data cannot be passed to an mm op"""
    return t.axis is not None and t.size() != t._data.size()


@register_qtensor_op([torch.ops.aten.bmm])
def bmm(op, input, other):
    if not isinstance(input, QTensor):
        return op(input, other.dequantize())
    if not isinstance(other, QTensor) or input.axis is not None:
        return op(input.dequantize(), other)
    if input.qtype != qint8 or other.qtype != qint8 or cannot_mm(other):
        return qfallback(op, input, other)
    # Cast data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = (input._scale * other._scale).to(torch.float32)
    return (out_data * out_scale).to(input._scale.dtype)


@register_qtensor_op([torch.ops.aten.mm])
def mm(op, input, other):
    if not isinstance(input, QTensor):
        if cannot_mm(other):
            return op(input, other.dequantize())
        return torch.ops.quanto.dqmm(input, other._data, other._scale)
    if not isinstance(other, QTensor) or input.axis is not None:
        return op(input.dequantize(), other)
    if input.qtype != qint8 or other.qtype != qint8 or cannot_mm(other):
        return qfallback(op, input, other)
    n, m = input.shape
    p = other.shape[-1]
    if (
        (
            input.device.type == "cuda"
            or (
                input.device.type == "cpu"
                and version.parse(torch.__version__).release >= version.parse("2.4.0").release
            )
        )
        and input.qtype == qint8
        and other.qtype == qint8
        and n > 16
        and n % 8 == 0
        and m % 8 == 0
        and p % 8 == 0
    ):
        # Use integer GEMM
        out_data = torch._int_mm(input._data, other._data)
    else:
        # Cast data to float32 and do the operation
        out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = (input._scale * other._scale).to(torch.float32)
    return (out_data * out_scale).to(input._scale.dtype)


@register_qtensor_op([torch.ops.aten.mul])
def mul(op, input, other):
    # If one of the multiplicands is a scalar, just multiply the scale
    if is_scalar(input):
        return QTensor(other.qtype, other.axis, other.size(), other.stride(), other._data, input * other._scale)
    if is_scalar(other):
        return QTensor(input.qtype, input.axis, input.size(), input.stride(), input._data, other * input._scale)
    return qfallback(op, input, other)


@register_qtensor_op([torch.ops.aten.relu])
def relu(op, input):
    if input.qtype.is_floating_point:
        # Relu is not supported for float8 types
        return qfallback(op, input)
    out_data = op(input._data)
    return QTensor(input.qtype, input.axis, input.size(), input.stride(), out_data, input._scale)


@register_qtensor_op([torch.ops.aten._softmax])
def _softmax(op, input, dim, half_to_float):
    # Softmax must be performed in float
    float_data = op(input.dequantize(), dim, half_to_float)
    # Since softmax is normalized, we know the optimal scale

    out_scale = torch.tensor(1 / dtype_info(input.qtype.dtype).max, dtype=input._scale.dtype).to(input.device)
    return QTensor.quantize(float_data, qtype=input.qtype, axis=None, group_size=None, scale=out_scale)


@register_qtensor_op([torch.ops.aten.stack])
def stack(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        # Only quantized tensors with identical scales can be stacked
        if (
            isinstance(t1, QTensor)
            and isinstance(t2, QTensor)
            and t1.axis is None
            and t2.axis is None
            and torch.equal(t1._scale, t2._scale)
            and t1.qtype == t2.qtype
        ):
            out_data = op([t1._data, t2._data], dim)
            return QTensor(t1.qtype, t1.axis, out_data.size(), out_data.stride(), out_data, t1._scale)
    return qfallback(inputs, dim)


@register_qtensor_op([torch.ops.aten.split])
def split(op, input, *args, **kwargs):
    if input.axis is not None:
        return qfallback(op, input, *args, **kwargs)
    out_datas = op(input._data, *args, **kwargs)
    return [
        QTensor(input.qtype, input.axis, input.size(), input.stride(), out_data, input._scale)
        for out_data in out_datas
    ]


@register_qtensor_op([torch.ops.aten.transpose])
def transpose(op, input, *args):
    if input.axis is not None:
        return op(input.dequantize(), *args)
    out_data = op(input._data, *args)
    out_size = out_data.size()
    out_stride = out_data.stride()
    out_scale = input._scale
    return QTensor(input.qtype, None, out_size, out_stride, out_data, out_scale)


@register_qtensor_op([torch.ops.aten.t])
def transpose2d(op, input):
    out_data = op(input._data)
    out_scale = input._scale
    out_axis = input.axis
    # Manually reverse size and stride because we cannot trust the out_data shape when using group-wise quantization
    dim0, dim1 = input.size()
    out_size = torch.Size([dim1, dim0])
    out_stride = input.stride()[::-1]
    if input.axis is not None:
        # We need to transpose also the scale
        out_scale = op(out_scale)
        out_axis = 0 if out_axis == -1 else -1
    return QTensor(input.qtype, out_axis, out_size, out_stride, out_data, out_scale)


@register_qtensor_op([torch.ops.aten.view, torch.ops.aten._unsafe_view])
def view(op, input, *shape):
    if input.axis is None:
        # The view is transparent for QTensor with scalar scales
        out_data = op(input._data, *shape)
        return QTensor(input.qtype, None, out_data.size(), out_data.stride(), out_data, input._scale)
    return qfallback(op, input, *shape)


@register_qtensor_op([torch.ops.aten.where])
def where(op, condition, input, other):
    if isinstance(condition, QTensor) or isinstance(other, QTensor):
        raise NotImplementedError
    float_data = op(condition, input.dequantize(), other)
    if input.axis is None:
        # We requantize with the input scale
        return QTensor.quantize(float_data, qtype=input.qtype, axis=None, group_size=None, scale=input._scale)
    return float_data
