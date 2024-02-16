import numbers
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, List, Optional

import torch

from . import QTensor, dtype_info, qfallback
from .qtype import qint8, qint32


__all__ = ["get_qtensor_op_dispatch", "register_qtensor_op"]


@dataclass
class QArg:
    """A simple class to describe the expected QTensor type of an argument."""

    index: int = 0
    axis: List[Any] = field(default_factory=[None])


@dataclass
class QOpDispatch:
    """A simple class to describe a quantized dispatched operation.

    It contains the quantized operation and the expected QTensor types
    for its arguments.
    """

    qop: Callable
    qargs: List[QArg] = field(default_factory=lambda: [])


_QTENSOR_OP_TABLE = {}


def register_qtensor_op(aten_ops: List[Callable], qargs: Optional[List[QArg]] = []):
    """
    Used for registering a new __torch_dispatch__ aten operation to QTensor.

    The code to register a new operation looks like:

    @register_qtensor_op(list_of_ops)
    def foo(op, *args, **kwargs):
        <implementation>
    """

    def wrapper(op):
        for aten_op in aten_ops:
            _QTENSOR_OP_TABLE[aten_op] = QOpDispatch(partial(op, aten_op), qargs)

    return wrapper


def get_qtensor_op_dispatch(aten_op):
    return _QTENSOR_OP_TABLE.get(aten_op, None)


def ensure_qtensor_inputs(*args, per_tensor=True):
    for arg in args:
        if not isinstance(arg, QTensor):
            return False
        if per_tensor and arg.axis is not None:
            return False
    return True


def is_scalar(t):
    return isinstance(t, numbers.Number) or type(t) == torch.Tensor and len(t.shape) == 0


@register_qtensor_op([torch.ops.aten._to_copy])
def _to_copy(op, t, dtype=None, **kwargs):
    # For data, ignore dtype and use the inner type instead
    out_data = op(t._data, dtype=t._data.dtype, **kwargs)
    # Apply the new dtype on the scale only
    out_scale = op(t._scale, dtype=dtype, **kwargs)
    return QTensor(t.qtype, out_data, out_scale)


@register_qtensor_op([torch.ops.aten.detach])
def detach(op, t):
    # Detach both data and scale
    out_data = op(t._data)
    out_scale = op(t._scale)
    return QTensor(t.qtype, out_data, out_scale)


@register_qtensor_op([torch.ops.aten.cat])
def cat(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        if (
            isinstance(t1, QTensor)
            and isinstance(t2, QTensor)
            and torch.equal(t1._scale, t2._scale)
            and t1.qtype == t2.qtype
        ):
            if t1.qtype.is_floating_point or t2.qtype.is_floating_point:
                # Cat is not supported for float8
                return qfallback(op, inputs, dim)
            # Only quantized tensors with identical scales can be concatenated
            out_data = op([t1._data, t2._data], dim)
            return QTensor(t1.qtype, out_data, t1._scale)
    return qfallback(op, inputs, dim)


@register_qtensor_op([torch.ops.aten.lt])
def lt(op, input, other):
    # Only quantized tensors with identical scales can be compared
    if isinstance(input, QTensor) and isinstance(other, QTensor) and torch.equal(input._scale, other._scale):
        return op(input._data, other._data)
    return qfallback(op, input, other)


@register_qtensor_op(
    [torch.ops.aten.addmm],
    qargs=[QArg(index=0, axis=[None, -1]), QArg(index=1, axis=[None]), QArg(index=2, axis=[None, -1])],
)
def addmm(op, input, mat1, mat2, beta=1, alpha=1):
    if alpha != 1 or beta != 1 or mat1.qtype != qint8 or mat2.qtype != qint8:
        return qfallback(op, input, mat1, mat2, beta=beta, alpha=alpha)
    # Do the operation with data cast to float32
    out_data = op(
        input._data.to(torch.float32),
        mat1._data.to(torch.float32),
        mat2._data.to(torch.float32),
        beta=beta,
        alpha=alpha,
    )
    out_scale = mat1._scale * mat2._scale
    return QTensor(qint32, out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.clone])
def clone(op, t, memory_format=torch.preserve_format):
    out_data = op(t._data, memory_format=memory_format)
    out_scale = op(t._scale, memory_format=memory_format)
    return QTensor(t.qtype, out_data, out_scale)


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
    return QTensor(input.qtype, input._data, op(input._scale, other))


@register_qtensor_op([torch.ops.aten.dot], qargs=[QArg(index=0, axis=[None]), QArg(index=1, axis=[None])])
def dot(op, input, other):
    if input.qtype != qint8 or other.qtype != qint8:
        return qfallback(op, input, other)
    # Cast data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(qint32, out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.neg])
def neg(op, input, *args, **kwargs):
    if input.qtype.is_floating_point:
        # Neg is not supported for float8
        return op(input.dequantize(), *args, **kwargs)
    out_data = op(input._data, *args, **kwargs)
    return QTensor(input.qtype, out_data, input._scale)


@register_qtensor_op(
    [
        torch.ops.aten.expand,
        torch.ops.aten.permute,
        torch.ops.aten.select,
        torch.ops.aten.slice,
        torch.ops.aten.unsqueeze,
    ],
    qargs=[QArg(index=0, axis=[None])],
)
def unary_type_agnostic_op(op, input, *args, **kwargs):
    # When quantization is per-tensor, rhese operations can be transparently applied
    # without modifying the scale.
    out_data = op(input._data, *args, **kwargs)
    return QTensor(input.qtype, out_data, input._scale)


@register_qtensor_op([torch.ops.aten.is_same_size])
def is_same_size(op, input, other):
    a = input._data if isinstance(input, QTensor) else input
    b = other._data if isinstance(other, QTensor) else other
    return op(a, b)


@register_qtensor_op([torch.ops.aten.linear])
def linear(op, input, weight, bias=None):
    if (
        not isinstance(input, QTensor)
        or input.axis is not None
        or not isinstance(weight, QTensor)
        or input.qtype != qint8
        or weight.qtype != qint8
        or (bias is not None and not isinstance(bias, QTensor))
    ):
        return qfallback(op, input, weight, bias=bias)
    # Cast int8 data to float32 and do the operation
    bias_data = bias._data.to(torch.float32) if bias is not None else None
    out_data = op(input._data.to(torch.float32), weight._data.to(torch.float32), bias_data)
    # The scalar input scale is broadcasted along all input dimensions
    input_scale = input._scale.view((1,) * input.ndim)
    # Weights are actually transposed inside the operation
    weight_scale = weight._scale.t()
    out_scale = input_scale * weight_scale
    return QTensor(qint32, out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.bmm], qargs=[QArg(index=0, axis=[None]), QArg(index=1, axis=[None, -1])])
def bmm(op, input, other):
    if input.qtype != qint8 or other.qtype != qint8:
        return qfallback(op, input, other)
    # Cast data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(qint32, out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.mm], qargs=[QArg(index=0, axis=[None]), QArg(index=1, axis=[None, -1])])
def mm(op, input, other):
    if input.qtype != qint8 or other.qtype != qint8:
        return qfallback(op, input, other)
    n, m = input.shape
    p = other.shape[-1]
    if (
        input.device.type == "cuda"
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
    out_scale = input._scale * other._scale
    return QTensor(qint32, out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.mul])
def mul(op, input, other):
    # If one of the multiplicands is a scalar, just multiply the scale
    if is_scalar(input):
        return QTensor(other.qtype, other._data, input * other._scale)
    if is_scalar(other):
        return QTensor(input.qtype, input._data, other * input._scale)
    if (
        not isinstance(input, QTensor)
        or not isinstance(other, QTensor)
        or input.qtype != qint8
        or other.qtype != qint8
    ):
        return qfallback(op, input, other)
    # Cast int8 data to int32 and do the operation
    out_data = op(input._data.to(torch.int32), other._data.to(torch.int32))
    out_scale = input._scale * other._scale
    return QTensor(qint32, out_data, out_scale)


@register_qtensor_op([torch.ops.aten.relu])
def relu(op, input):
    if input.qtype.is_floating_point:
        # Relu is not supported for float8 types
        return qfallback(op, input)
    out_data = op(input._data)
    return QTensor(input.qtype, out_data, input._scale)


@register_qtensor_op([torch.ops.aten._softmax])
def _softmax(op, input, dim, half_to_float):
    # Softmax must be performed in float
    float_data = op(input.dequantize(), dim, half_to_float)
    # Since softmax is normalized, we know the optimal scale

    out_scale = torch.tensor(1 / dtype_info(input.qtype.dtype).max, dtype=input._scale.dtype).to(input.device)
    return QTensor.quantize(float_data, input.qtype, out_scale)


@register_qtensor_op([torch.ops.aten.stack])
def stack(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        if (
            isinstance(t1, QTensor)
            and isinstance(t2, QTensor)
            and torch.equal(t1._scale, t2._scale)
            and t1.qtype == t2.qtype
        ):
            # Only quantized tensors with identical scales can be stacked
            out_data = op([t1._data, t2._data], dim)
            return QTensor(t1.qtype, out_data, t1._scale)
    return qfallback(inputs, dim)


@register_qtensor_op([torch.ops.aten.split])
def split(op, input, *args, **kwargs):
    if input.axis is not None:
        return qfallback(op, input, *args, **kwargs)
    out_datas = op(input._data, *args, **kwargs)
    return [QTensor(input.qtype, out_data, input._scale) for out_data in out_datas]


@register_qtensor_op([torch.ops.aten.transpose, torch.ops.aten.t])
def transpose(op, input, *args):
    out_data = op(input._data, *args)
    out_scale = input._scale
    if input.axis is not None:
        # We need to transpose also the scale
        out_scale = op(out_scale, *args)
    return QTensor(input.qtype, out_data, out_scale)


@register_qtensor_op([torch.ops.aten.view, torch.ops.aten._unsafe_view], qargs=[QArg(index=0, axis=[None, -1])])
def view(op, input, *shape):
    out_data = op(input._data, *shape)
    if input.axis is None:
        # The view is transparent for QTensor with scalar scales
        return QTensor(input.qtype, out_data, input._scale)
    # The tensor is quantized along the last axis
    assert input.axis == -1
    # We can only perform the view if the last axis is not modified
    if input._scale.shape[-1] == out_data.shape[-1]:
        out_scale_shape = (1,) * (out_data.ndim - 1) + (input._scale.shape[-1],)
        out_scale = input._scale.view(out_scale_shape)
        return QTensor(input.qtype, out_data, out_scale)
    return qfallback(op, input, *shape)


@register_qtensor_op([torch.ops.aten.where])
def where(op, condition, input, other):
    if isinstance(condition, QTensor) or isinstance(other, QTensor):
        raise NotImplementedError
    float_data = op(condition, input.dequantize(), other)
    # We requantize with the input scale
    return QTensor.quantize(float_data, input.qtype, input._scale)
