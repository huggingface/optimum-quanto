import numbers
from functools import partial

import torch

from . import QTensor


__all__ = ["get_qtensor_op", "register_qtensor_op"]


_QTENSOR_OP_TABLE = {}


def register_qtensor_op(aten_ops):
    """
    Used for registering a new __torch_dispatch__ aten operation to QTensor

    The code to register a new operation looks like:

    @register_qtensor_op(list_of_ops)
    def foo(op, *args, **kwargs):
        <implementation>
    """

    def wrapper(op):
        for aten_op in aten_ops:
            _QTENSOR_OP_TABLE[aten_op] = partial(op, aten_op)

    return wrapper


def get_qtensor_op(aten_op):
    return _QTENSOR_OP_TABLE.get(aten_op, None)


def dequantize(*args):
    return [arg.dequantize() if isinstance(arg, QTensor) else arg for arg in args]


def is_scalar(t):
    return isinstance(t, numbers.Number) or type(t) == torch.Tensor and len(t.shape) == 0


@register_qtensor_op([torch.ops.aten._to_copy])
def _to_copy(op, t, dtype=None, **kwargs):
    # Ignore dtype and use the inner data tensors dtypes instead
    out_data = op(t._data, dtype=t._data.dtype, **kwargs)
    out_scale = op(t._scale, dtype=t._scale.dtype, **kwargs)
    return QTensor(out_data, out_scale)


@register_qtensor_op([torch.ops.aten.argmax])
def argmax(op, input, *args, **kwargs):
    # We just return the argmax for the data
    return op(input._data, *args, **kwargs)


@register_qtensor_op([torch.ops.aten.detach])
def detach(op, t):
    # Detach both data and scale
    out_data = op(t._data)
    out_scale = op(t._scale)
    return QTensor(out_data, out_scale)


@register_qtensor_op([torch.ops.aten.add])
def add(op, input, other):
    # Only quantized tensors with identical scales cannot be added
    if isinstance(input, QTensor) and isinstance(other, QTensor) and torch.equal(input._scale, other._scale):
        # We need to perform the operation in int16 because it might overflow
        out_data = op(input._data.to(torch.int16), other._data.to(torch.int16))
        out_scale = input._scale
        return QTensor(out_data, out_scale)
    return op(*dequantize(input, other))


@register_qtensor_op([torch.ops.aten.cat])
def cat(op, inputs, dim):
    if len(inputs) == 2:
        t1, t2 = inputs
        if isinstance(t1, QTensor) and isinstance(t2, QTensor) and torch.equal(t1._scale, t2._scale):
            # Only quantized tensors with identical scales can be concatenated
            out_data = op([t1._data, t2._data], dim)
            return QTensor(out_data, t1._scale)
    return op(*dequantize(inputs), dim)


@register_qtensor_op([torch.ops.aten.lt])
def lt(op, input, other):
    # Only quantized tensors with identical scales can be compared
    if isinstance(input, QTensor) and isinstance(other, QTensor) and torch.equal(input._scale, other._scale):
        return op(input._data, other._data)
    return op(*dequantize(input, other))


@register_qtensor_op([torch.ops.aten.addmm])
def addmm(op, input, mat1, mat2, beta=1, alpha=1):
    # Do the operation with int8 cast to float32
    out_data = op(
        input._data.to(torch.float32),
        mat1._data.to(torch.float32),
        mat2._data.to(torch.float32),
        beta=beta,
        alpha=alpha,
    )
    out_scale = mat1._scale * mat2._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.copy_])
def copy_(op, dest, src):
    dest._data = op(dest._data, src._data)
    dest._scale = op(dest._scale, src._scale)
    return dest


@register_qtensor_op([torch.ops.aten.div])
def div(op, input, other):
    if not is_scalar(other):
        raise NotImplementedError()
    # We just divide the scale
    return QTensor(input._data, op(input._scale, other))


@register_qtensor_op([torch.ops.aten.dot])
def dot(op, input, other):
    # Cast int8 data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.expand, torch.ops.aten.permute, torch.ops.aten.select, torch.ops.aten.slice])
def unary_type_agnostic_op(op, input, *args, **kwargs):
    out_data = op(input._data, *args, **kwargs)
    return QTensor(out_data, input._scale)


@register_qtensor_op([torch.ops.aten.is_same_size])
def is_same_size(op, input, other):
    a = input._data if isinstance(input, QTensor) else input
    b = other._data if isinstance(other, QTensor) else other
    return op(a, b)


@register_qtensor_op([torch.ops.aten.gelu, torch.ops.aten.masked_fill])
def unary_unsupported_op(op, input, *args, **kwargs):
    # Not supported: dequantize
    return op(input.dequantize(), *args, **kwargs)


@register_qtensor_op([torch.ops.aten.bmm, torch.ops.aten.mm])
def mm(op, input, other):
    if not isinstance(input, QTensor) or not isinstance(other, QTensor):
        return op(*dequantize(input, other))
    # Cast int8 data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.mul])
def mul(op, input, other):
    # If one of the multiplicands is a scalar, just multiply the scale
    if is_scalar(input):
        return QTensor(other._data, input * other._scale)
    if is_scalar(other):
        return QTensor(input._data, other * input._scale)
    if not isinstance(input, QTensor) or not isinstance(other, QTensor):
        return op(*dequantize(input, other))
    # Cast int8 data to int32 and do the operation
    out_data = op(input._data.to(torch.int32), other._data.to(torch.int32))
    out_scale = input._scale * other._scale
    return QTensor(out_data, out_scale)


@register_qtensor_op([torch.ops.aten.relu])
def relu(op, input):
    out_data = op(input._data)
    return QTensor(out_data, input._scale)


@register_qtensor_op([torch.ops.aten._softmax])
def _softmax(op, input, dim, half_to_float):
    # Softmax must be performed in float
    out_data = op(input.dequantize(), dim, half_to_float)
    # Since softmax is normalized, we know the optimal scale
    out_scale = torch.tensor(1 / torch.iinfo(input._data.dtype).max, dtype=input._scale.dtype)
    return QTensor.quantize(out_data, input._data.dtype, out_scale)


@register_qtensor_op([torch.ops.aten.split])
def split(op, input, *args, **kwargs):
    out_datas = op(input._data, *args, **kwargs)
    return [QTensor(out_data, input._scale) for out_data in out_datas]


@register_qtensor_op([torch.ops.aten.transpose, torch.ops.aten.t])
def transpose(op, input, *args):
    # Transpose is not supported if the tensor is per-axis
    assert len(input._scale.shape) == 0
    out_data = op(input._data, *args)
    return QTensor(out_data, input._scale)


@register_qtensor_op([torch.ops.aten.view, torch.ops.aten._unsafe_view])
def view(op, input, *shape):
    out_data = op(input._data, *shape)
    return QTensor(out_data, input._scale)


@register_qtensor_op([torch.ops.aten._softmax_backward_data])
def _softmax_backward_data(op, grad, output, dim, input_dtype):
    return op(grad, output.dequantize(), dim, input_dtype)


@register_qtensor_op([torch.ops.aten.threshold_backward])
def threshold_backward(op, grad, output, threshold):
    return op(grad, output.dequantize(), threshold)


@register_qtensor_op([torch.ops.aten.where])
def where(op, condition, input, other):
    if isinstance(condition, QTensor) or isinstance(other, QTensor):
        raise NotImplementedError
    float_data = op(condition, input.dequantize(), other)
    # We requantize with the input scale
    return QTensor.quantize(float_data, input._data.dtype, input._scale)
