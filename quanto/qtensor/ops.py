import numbers
from functools import partial

import torch

from . import QTensor, dtype_info


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


def dequantized_op(func, *args, **kwargs):
    def dequantize(x):
        if isinstance(x, list):
            return [dequantize(y) for y in x]
        if isinstance(x, tuple):
            return (dequantize(y) for y in x)
        return x.dequantize() if isinstance(x, QTensor) else x

    dq_args = [dequantize(arg) for arg in args]
    dq_kwargs = {}
    for name, kwarg in kwargs.items():
        dq_kwargs[name] = dequantize(kwarg)
    return func(*dq_args, **dq_kwargs)


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
    out_data = op(t._data, dtype=t.itype, **kwargs)
    # Apply the new dtype on the scale only
    out_scale = op(t._scale, dtype=dtype, **kwargs)
    return QTensor(out_data, out_scale)


@register_qtensor_op([torch.ops.aten.argmax])
def argmax(op, input, *args, **kwargs):
    if input.axis is not None:
        # If we have different scales we need to dequantize first
        return dequantized_op(op, input, *args, **kwargs)
    if input.itype.is_floating_point:
        # Argmax is not supported for float8
        return dequantized_op(op, input, *args, **kwargs)
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
    return dequantized_op(op, input, other)


@register_qtensor_op([torch.ops.aten.cat])
def cat(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        if isinstance(t1, QTensor) and isinstance(t2, QTensor) and torch.equal(t1._scale, t2._scale):
            if t1.itype.is_floating_point or t2.itype.is_floating_point:
                # Cat is not supported for float8
                return dequantized_op(op, inputs, dim)
            # Only quantized tensors with identical scales can be concatenated
            out_data = op([t1._data, t2._data], dim)
            return QTensor(out_data, t1._scale)
    return dequantized_op(op, inputs, dim)


@register_qtensor_op([torch.ops.aten.lt])
def lt(op, input, other):
    # Only quantized tensors with identical scales can be compared
    if isinstance(input, QTensor) and isinstance(other, QTensor) and torch.equal(input._scale, other._scale):
        return op(input._data, other._data)
    return dequantized_op(op, input, other)


@register_qtensor_op([torch.ops.aten.addmm])
def addmm(op, input, mat1, mat2, beta=1, alpha=1):
    if (
        alpha != 1
        or beta != 1
        or not ensure_qtensor_inputs(input, mat1, mat2, per_tensor=False)
        or mat1.axis is not None
    ):
        return dequantized_op(op, input, mat1, mat2, beta=beta, alpha=alpha)
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


@register_qtensor_op([torch.ops.aten.clone])
def clone(op, t, memory_format=torch.preserve_format):
    out_data = op(t._data, memory_format=memory_format)
    out_scale = op(t._scale, memory_format=memory_format)
    return QTensor(out_data, out_scale)


@register_qtensor_op([torch.ops.aten.copy_])
def copy_(op, dest, src):
    dest._data = op(dest._data, src._data)
    dest._scale = op(dest._scale, src._scale)
    return dest


@register_qtensor_op([torch.ops.aten.div])
def div(op, input, other):
    if not is_scalar(other):
        return op(input.dequantize(), other)
    # We just divide the scale
    return QTensor(input._data, op(input._scale, other))


@register_qtensor_op([torch.ops.aten.dot])
def dot(op, input, other):
    if not ensure_qtensor_inputs(input, other):
        return dequantized_op(op, input, other)
    # Cast int8 data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.neg])
def neg(op, input, *args, **kwargs):
    if input.itype.is_floating_point:
        # Neg is not supported for float8
        return op(input.dequantize(), *args, **kwargs)
    out_data = op(input._data, *args, **kwargs)
    return QTensor(out_data, input._scale)


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
    if not ensure_qtensor_inputs(input):
        return op(input.dequantize(), *args, **kwargs)
    # These operations can be transparently applied on the underlying integer tensor,
    # without modifying the scale.
    out_data = op(input._data, *args, **kwargs)
    return QTensor(out_data, input._scale)


@register_qtensor_op([torch.ops.aten.is_same_size])
def is_same_size(op, input, other):
    a = input._data if isinstance(input, QTensor) else input
    b = other._data if isinstance(other, QTensor) else other
    return op(a, b)


@register_qtensor_op([torch.ops.aten.gelu, torch.ops.aten.masked_fill, torch.ops.aten.pow, torch.ops.aten.silu])
def unary_unsupported_op(op, input, *args, **kwargs):
    # Not supported: dequantize
    return op(input.dequantize(), *args, **kwargs)


@register_qtensor_op([torch.ops.aten.linear])
def linear(op, input, weight, bias=None):
    if (
        not isinstance(input, QTensor)
        or input.axis is not None
        or not isinstance(weight, QTensor)
        or (bias is not None and not isinstance(bias, QTensor))
    ):
        return dequantized_op(op, input, weight, bias=bias)
    # Cast int8 data to float32 and do the operation
    bias_data = bias._data.to(torch.float32) if bias is not None else None
    out_data = op(input._data.to(torch.float32), weight._data.to(torch.float32), bias_data)
    # The scalar input scale is broadcasted along all input dimensions
    input_scale = input._scale.view((1,) * input.ndim)
    # Weights are actually transposed inside the operation
    weight_scale = weight._scale.t()
    out_scale = input_scale * weight_scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.bmm])
def bmm(op, input, other):
    if not ensure_qtensor_inputs(input, other, per_tensor=False) or input.axis is not None:
        # Matric multiplication is only supported between a per-tensor QTensor and a QTensor
        return dequantized_op(op, input, other)
    # Cast int8 data to float32 and do the operation
    out_data = op(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.mm])
def mm(op, input, other):
    if not ensure_qtensor_inputs(input, other, per_tensor=False) or input.axis is not None:
        # Matric multiplication is only supported between a per-tensor QTensor and a QTensor
        return dequantized_op(op, input, other)
    n, m = input.shape
    p = other.shape[-1]
    if (
        input.device.type == "cuda"
        and input.itype == torch.int8
        and other.itype == torch.int8
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
    return QTensor(out_data.to(torch.int32), out_scale)


@register_qtensor_op([torch.ops.aten.mul])
def mul(op, input, other):
    # If one of the multiplicands is a scalar, just multiply the scale
    if is_scalar(input):
        return QTensor(other._data, input * other._scale)
    if is_scalar(other):
        return QTensor(input._data, other * input._scale)
    if not isinstance(input, QTensor) or not isinstance(other, QTensor):
        return dequantized_op(op, input, other)
    # Cast int8 data to int32 and do the operation
    out_data = op(input._data.to(torch.int32), other._data.to(torch.int32))
    out_scale = input._scale * other._scale
    return QTensor(out_data, out_scale)


@register_qtensor_op([torch.ops.aten.relu])
def relu(op, input):
    if input.itype.is_floating_point:
        # Relu is not supported for float8 types
        return dequantized_op(op, input)
    out_data = op(input._data)
    return QTensor(out_data, input._scale)


@register_qtensor_op([torch.ops.aten._softmax])
def _softmax(op, input, dim, half_to_float):
    # Softmax must be performed in float
    out_data = op(input.dequantize(), dim, half_to_float)
    # Since softmax is normalized, we know the optimal scale

    out_scale = torch.tensor(1 / dtype_info(input.itype).max, dtype=input.dtype).to(input.device)
    return QTensor.quantize(out_data, input.itype, out_scale)


@register_qtensor_op([torch.ops.aten.stack])
def stack(op, inputs, dim=0):
    if len(inputs) == 2:
        t1, t2 = inputs
        if isinstance(t1, QTensor) and isinstance(t2, QTensor) and torch.equal(t1._scale, t2._scale):
            # Only quantized tensors with identical scales can be stacked
            out_data = op([t1._data, t2._data], dim)
            return QTensor(out_data, t1._scale)
    return dequantized_op(inputs, dim)


@register_qtensor_op([torch.ops.aten.split])
def split(op, input, *args, **kwargs):
    out_datas = op(input._data, *args, **kwargs)
    return [QTensor(out_data, input._scale) for out_data in out_datas]


@register_qtensor_op([torch.ops.aten.transpose, torch.ops.aten.t])
def transpose(op, input, *args):
    out_data = op(input._data, *args)
    out_scale = input._scale
    if input.axis is not None:
        # We need to transpose also the scale
        out_scale = op(out_scale, *args)
    return QTensor(out_data, out_scale)


@register_qtensor_op([torch.ops.aten.view, torch.ops.aten._unsafe_view])
def view(op, input, *shape):
    out_data = op(input._data, *shape)
    if ensure_qtensor_inputs(input):
        return QTensor(out_data, input._scale)
    # We only support the simple case where the tensor is quantized along the
    # last axis that is not modified by the view
    if input.axis == -1 and input._scale.shape[-1] == out_data.shape[-1]:
        out_scale_shape = (1,) * (out_data.ndim - 1) + (input._scale.shape[-1],)
        out_scale = input._scale.view(out_scale_shape)
        return QTensor(out_data, out_scale)
    return dequantized_op(op, input, *shape)


@register_qtensor_op([torch.ops.aten._softmax_backward_data])
def _softmax_backward_data(op, grad, output, dim, input_dtype):
    return op(grad, output.dequantize(), dim, input_dtype)


@register_qtensor_op([torch.ops.aten.threshold_backward])
def threshold_backward(op, grad, output, threshold):
    return op(grad, output.dequantize(), threshold)


@register_qtensor_op([torch.ops.aten.linear_backward])
def linear_backward(op, *args, **kwargs):
    return dequantized_op(op, *args, **kwargs)


@register_qtensor_op([torch.ops.aten.where])
def where(op, condition, input, other):
    if isinstance(condition, QTensor) or isinstance(other, QTensor):
        raise NotImplementedError
    float_data = op(condition, input.dequantize(), other)
    # We requantize with the input scale
    return QTensor.quantize(float_data, input.itype, input._scale)
