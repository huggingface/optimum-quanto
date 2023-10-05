import torch
from torch.autograd import Function


def scale_max(base, int_dtype):
    return torch.max(torch.abs(base)) / torch.iinfo(int_dtype).max


class LinearQuantizer(Function):
    @staticmethod
    def forward(ctx, base, int_dtype=torch.int8, scale=None):
        if scale is None:
            scale = scale_max(base, int_dtype)
        iinfo = torch.iinfo(int_dtype)
        data = torch.clamp(torch.round(base / scale), min=iinfo.min, max=iinfo.max).to(int_dtype)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.
        return QuantizedTensor(data, scale)

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
        return QuantizedTensor(data, scale)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, requantization is a no-op
        return gO, None, None


def q_to_copy(func, t, dtype=None, **kwargs):
    # Ignore dtype and use the inner data tensors dtypes instead
    out_data = func(t._data, dtype=t._data.dtype, **kwargs)
    out_scale = func(t._scale, dtype=t._scale.dtype, **kwargs)
    return QuantizedTensor(out_data, out_scale)


def q_detach(func, t):
    # Detach both data and scale
    out_data = func(t._data)
    out_scale = func(t._scale)
    return QuantizedTensor(out_data, out_scale)


def q_add(func, input, other, alpha=1, out=None):
    if alpha != 1 or out is not None:
        raise ValueError("alpha and out parameters are not supported for quantized {func}.")
    if not torch.equal(input._scale, other._scale):
        raise ValueError("Quantized tensors with different scales cannot be added.")
    # We need to perform the operation in int16 because it might overflow
    out_data = func(input._data.to(torch.int16), other._data.to(torch.int16))
    out_scale = input._scale
    return QuantizedTensor(out_data, out_scale)


def q_addmm(func, input, mat1, mat2, beta=1, alpha=1):
    # Do the operation with int8 cast to float32
    out_data = func(
        input._data.to(torch.float32),
        mat1._data.to(torch.float32),
        mat2._data.to(torch.float32),
        beta=beta,
        alpha=alpha,
    )
    out_scale = mat1._scale * mat2._scale
    return QuantizedTensor(out_data.to(torch.int32), out_scale)


def q_copy(func, dest, src):
    dest._data = func(dest._data, src._data)
    dest._scale = func(dest._scale, src._scale)
    return dest


def q_dot(func, input, other):
    # Cast int8 data to float32 and do the operation
    out_data = func(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QuantizedTensor(out_data.to(torch.int32), out_scale)


def q_is_same_size(func, input, other):
    a = input._data if isinstance(input, QuantizedTensor) else input
    b = other._data if isinstance(other, QuantizedTensor) else other
    return func(a, b)


def q_mm(func, input, other):
    if not isinstance(input, QuantizedTensor) or not isinstance(other, QuantizedTensor):
        return func(input.dequantize(), other.dequantize())
    # Cast int8 data to float32 and do the operation
    out_data = func(input._data.to(torch.float32), other._data.to(torch.float32))
    out_scale = input._scale * other._scale
    return QuantizedTensor(out_data.to(torch.int32), out_scale)


def q_mul(func, input, other):
    if not isinstance(input, QuantizedTensor) or not isinstance(other, QuantizedTensor):
        return func(input.dequantize(), other.dequantize())
    # Cast int8 data to int32 and do the operation
    out_data = func(input._data.to(torch.int32), other._data.to(torch.int32))
    out_scale = input._scale * other._scale
    return QuantizedTensor(out_data, out_scale)


def q_relu(func, input):
    out_data = func(input._data)
    return QuantizedTensor(out_data, input._scale)


def q_softmax(func, input, dim, half_to_float):
    # Softmax must be performed in float
    out_data = func(input.dequantize(), dim, half_to_float)
    # Since softmax is normalized, we know the optimal scale
    out_scale = torch.tensor(1 / torch.iinfo(input._data.dtype).max, dtype=input._scale.dtype)
    return QuantizedTensor.quantize(out_data, input._data.dtype, out_scale)


def q_transpose(func, input):
    # Transpose is not supported if the tensor is per-axis
    assert len(input._scale.shape) == 0
    out_data = func(input._data)
    return QuantizedTensor(out_data, input._scale)


def q_view(func, input, *shape):
    out_data = func(input._data, *shape)
    return QuantizedTensor(out_data, input._scale)


def q_softmax_backward_data(func, grad, output, dim, input_dtype):
    return func(grad, output.dequantize(), dim, input_dtype)


def q_threshold_backward(func, grad, output, threshold):
    return func(grad, output.dequantize(), threshold)


quantized_dispatch = {
    torch.ops.aten.add.Tensor: q_add,
    torch.ops.aten.addmm.default: q_addmm,
    torch.ops.aten.bmm.default: q_mm,
    torch.ops.aten.copy_.default: q_copy,
    torch.ops.aten.detach.default: q_detach,
    torch.ops.aten.dot.default: q_dot,
    torch.ops.aten.is_same_size.default: q_is_same_size,
    torch.ops.aten.mm.default: q_mm,
    torch.ops.aten.mul.Tensor: q_mul,
    torch.ops.aten.relu.default: q_relu,
    torch.ops.aten._softmax.default: q_softmax,
    torch.ops.aten.t.default: q_transpose,
    torch.ops.aten.view.default: q_view,
    torch.ops.aten._to_copy.default: q_to_copy,
    torch.ops.aten._unsafe_view.default: q_view,
    torch.ops.aten._softmax_backward_data.default: q_softmax_backward_data,
    torch.ops.aten.threshold_backward.default: q_threshold_backward,
}


class QuantizedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, scale, requires_grad=False):
        # This constructor can ONLY create leaf Tensors wrt autograd.
        # Use QuantizedTensor.from_tensor(t) to get a non-leaf Tensor wrt autograd.
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), dtype=scale.dtype, requires_grad=requires_grad)

    def __init__(self, data, scale, requires_grad=False):
        self._data = data
        self._scale = scale

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):  # Zero out missing values for printing
        autograd_info = (
            f", grad_fn={self.grad_fn}" if self.grad_fn else ", requires_grad=True" if self.requires_grad else ""
        )
        return f"QuantizedTensor({self._data}, scale={self._scale}, public_dtype={self.dtype}{autograd_info})"

    @classmethod
    def quantize(cls, base, int_dtype=torch.int8, scale=None):
        """Differentiable quantization function"""
        return LinearQuantizer.apply(base, int_dtype, scale)

    def dequantize(self):
        """Differentiable dequantization function"""
        return Dequantizer.apply(self)

    def rescale(self, int_dtype=torch.int8, scale=None):
        """Differentiable requantization function"""
        return ReQuantizer.apply(self, int_dtype, scale)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func in quantized_dispatch:
            dispatch = quantized_dispatch[func]
            return dispatch(func, *args, **kwargs)
        raise ValueError(f"{func} is no supported for {args}")

    @property
    def device(self):
        return self._data.device
