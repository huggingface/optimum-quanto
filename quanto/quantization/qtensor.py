from functools import partial

import torch
from torch.autograd import Function


class Quantizer(Function):
    """A standard affine quantizer.

    If the quantization scale is not specified, then it uses the optimal scale
    for the base tensor value range.
    """

    @staticmethod
    def forward(ctx, base, int_dtype=torch.int8, scale=None):
        iinfo = torch.iinfo(int_dtype)
        if scale is None:
            scale = torch.max(torch.abs(base)) / torch.iinfo(int_dtype).max
        data = torch.clamp(torch.round(base / scale), min=iinfo.min, max=iinfo.max).to(int_dtype)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.
        return QTensor(data, scale)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None


class Dequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        return t._scale * t._data

    @staticmethod
    def backward(ctx, gO):
        # For autograd, dequantization is a no-op
        return gO


class ReQuantizer(Function):
    @staticmethod
    def forward(ctx, base, int_dtype=torch.int8, scale=None):
        dst_iinfo = torch.iinfo(int_dtype)
        if scale is None:
            if int_dtype == base._data.dtype:
                return base
            # Assuming the base scale is correct, simply project to the target integer range
            src_iinfo = torch.iinfo(base._data.dtype)
            int_rescale = dst_iinfo.max / src_iinfo.max
            scale = base._scale * src_iinfo.max / dst_iinfo.max
        else:
            # It is up to the caller to make sure the scale is consistent with the target int dtype
            int_rescale = base._scale / scale
        data = torch.clamp(torch.round(base._data * int_rescale), min=dst_iinfo.min, max=dst_iinfo.max).to(int_dtype)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.
        return QTensor(data, scale)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, requantization is a no-op
        return gO, None, None


_QTENSOR_DISPATCH_TABLE = {}


def register_dispatch(aten_ops):
    """
    Used for registering a new __torch_dispatch__ function to QTensor
    Called via _QTENSOR_DISPATCH_TABLE[func](func, *args, **kwargs)

    The code to register a new function looks like:

    @register_dispatch(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(func):
        for aten_op in aten_ops:
            _QTENSOR_DISPATCH_TABLE[aten_op] = partial(func, aten_op)

    return wrapper


def dequantize(*args):
    return [arg.dequantize() if isinstance(arg, QTensor) else arg for arg in args]


@register_dispatch([torch.ops.aten._to_copy])
def _to_copy(func, t, dtype=None, **kwargs):
    # Ignore dtype and use the inner data tensors dtypes instead
    out_data = func(t._data, dtype=t._data.dtype, **kwargs)
    out_scale = func(t._scale, dtype=t._scale.dtype, **kwargs)
    return QTensor(out_data, out_scale)


@register_dispatch([torch.ops.aten.detach])
def detach(func, t):
    # Detach both data and scale
    out_data = func(t._data)
    out_scale = func(t._scale)
    return QTensor(out_data, out_scale)


@register_dispatch([torch.ops.aten.add])
def add(func, input, other, alpha=1, out=None):
    if alpha != 1 or out is not None:
        raise ValueError("alpha and out parameters are not supported for quantized {func}.")
    if not torch.equal(input._scale, other._scale):
        # Quantized tensors with different scales cannot be added
        return func(input.dequantize(), other.dequantize())
    # We need to perform the operation in int16 because it might overflow
    out_data = func(input._data.to(torch.int16), other._data.to(torch.int16))
    out_scale = input._scale
    return QTensor(out_data, out_scale)


@register_dispatch([torch.ops.aten.addmm])
def addmm(func, input, mat1, mat2, beta=1, alpha=1):
    # Do the operation with int8 cast to float32
    out_data = func(
        input._data.to(torch.float32),
        mat1._data.to(torch.float32),
        mat2._data.to(torch.float32),
        beta=beta,
        alpha=alpha,
    )
    out_scale = mat1._scale * mat2._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_dispatch([torch.ops.aten.copy_])
def copy_(func, dest, src):
    dest._data = func(dest._data, src._data)
    dest._scale = func(dest._scale, src._scale)
    return dest


@register_dispatch([torch.ops.aten.div])
def div(func, input, other):
    if not isinstance(other, float):
        raise NotImplementedError()
    # We just divide the scale
    return QTensor(input._data, func(input._scale, other))


@register_dispatch([torch.ops.aten.dot])
def dot(func, input, other):
    # Cast int8 data to float32 and do the operation
    out_data = func(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_dispatch([torch.ops.aten.is_same_size])
def is_same_size(func, input, other):
    a = input._data if isinstance(input, QTensor) else input
    b = other._data if isinstance(other, QTensor) else other
    return func(a, b)


@register_dispatch([torch.ops.aten.bmm, torch.ops.aten.mm])
def mm(func, input, other):
    if not isinstance(input, QTensor) or not isinstance(other, QTensor):
        return func(*dequantize(input, other))
    # Cast int8 data to float32 and do the operation
    out_data = func(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QTensor(out_data.to(torch.int32), out_scale)


@register_dispatch([torch.ops.aten.mul])
def mul(func, input, other):
    if not isinstance(input, QTensor) or not isinstance(other, QTensor):
        return func(*dequantize(input, other))
    # Cast int8 data to int32 and do the operation
    out_data = func(input._data.to(torch.int32), other._data.to(torch.int32))
    out_scale = input._scale * other._scale
    return QTensor(out_data, out_scale)


@register_dispatch([torch.ops.aten.relu])
def relu(func, input):
    out_data = func(input._data)
    return QTensor(out_data, input._scale)


@register_dispatch([torch.ops.aten._softmax])
def _softmax(func, input, dim, half_to_float):
    # Softmax must be performed in float
    out_data = func(input.dequantize(), dim, half_to_float)
    # Since softmax is normalized, we know the optimal scale
    out_scale = torch.tensor(1 / torch.iinfo(input._data.dtype).max, dtype=input._scale.dtype)
    return QTensor.quantize(out_data, input._data.dtype, out_scale)


@register_dispatch([torch.ops.aten.transpose, torch.ops.aten.t])
def transpose(func, input, *args):
    # Transpose is not supported if the tensor is per-axis
    assert len(input._scale.shape) == 0
    out_data = func(input._data, *args)
    return QTensor(out_data, input._scale)


@register_dispatch([torch.ops.aten.view, torch.ops.aten._unsafe_view])
def view(func, input, *shape):
    out_data = func(input._data, *shape)
    return QTensor(out_data, input._scale)


@register_dispatch([torch.ops.aten._softmax_backward_data])
def _softmax_backward_data(func, grad, output, dim, input_dtype):
    return func(grad, output.dequantize(), dim, input_dtype)


@register_dispatch([torch.ops.aten.threshold_backward])
def threshold_backward(func, grad, output, threshold):
    return func(grad, output.dequantize(), threshold)


class QTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, scale, requires_grad=False):
        # This constructor can ONLY create leaf Tensors wrt autograd.
        # Use QTensor.from_tensor(t) to get a non-leaf Tensor wrt autograd.
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), dtype=scale.dtype, requires_grad=requires_grad)

    def __init__(self, data, scale, requires_grad=False):
        self._data = data
        self._scale = scale

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):  # Zero out missing values for printing
        autograd_info = (
            f", grad_fn={self.grad_fn}" if self.grad_fn else ", requires_grad=True" if self.requires_grad else ""
        )
        return f"QTensor({self._data}, scale={self._scale}, public_dtype={self.dtype}{autograd_info})"

    @classmethod
    def quantize(cls, base, int_dtype=torch.int8, scale=None):
        """Differentiable quantization function"""
        return Quantizer.apply(base, int_dtype, scale)

    def dequantize(self):
        """Differentiable dequantization function"""
        return Dequantizer.apply(self)

    def rescale(self, int_dtype=torch.int8, scale=None):
        """Differentiable requantization function"""
        return ReQuantizer.apply(self, int_dtype, scale)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Do not use directly func, but rather its overload
        func = func.overloadpacket
        if func in _QTENSOR_DISPATCH_TABLE:
            dispatch = _QTENSOR_DISPATCH_TABLE[func]
            return dispatch(*args, **kwargs)
        # Identify the types of the args
        types = [type(arg).__name__ for arg in args]
        raise ValueError(f"{func} {types} is no supported for QTensor.")

    @property
    def device(self):
        return self._data.device

    def numpy(self):
        return self.dequantize().cpu().numpy()
