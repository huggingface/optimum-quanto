from typing import Optional

import torch
from torch.autograd import Function


__all__ = ["absmax_scale", "QTensor"]


def absmax_scale(
    base: torch.Tensor, int_dtype: torch.Tensor.dtype = torch.int8, axis: Optional[int] = None
) -> torch.Tensor:
    """Evaluate the quantization scale using the absmax algorithm.

    The Absolute Maximum quantization algorithm is a symmetrical quantization
    algorithm where the scale corresponds to the maximum absolute value of the
    base divided by the highest positive integer value for the target integer
    representation.

    Args:
        base (`torch.Tensor`): the base tensor on which the scale will be applied.
        int_dtype (`torch.Tensor.dtype`): the target integer dtype for quantization.
        axis (`int`): the index of the axis to preserve, or -1 for the last one.
            Defaults to None to reduce all axis.

    Returns:
        `torch.Tensor`: a scale tensor of the same dtype as the base.
    """
    abs_base = torch.abs(base)
    if axis is None:
        qranges = torch.max(abs_base)
    else:
        dim = list(range(base.ndim))
        if axis == -1:
            dim = dim[:-1]
        else:
            dim.remove(axis)
        qranges = torch.amax(torch.abs(base), dim=dim, keepdim=True)
    return qranges / torch.iinfo(int_dtype).max


class Quantizer(Function):
    """A standard symmetric quantizer.

    If the quantization scale is not specified, it uses the absmax scale for the base tensor.
    """

    @staticmethod
    def forward(ctx, base, int_dtype: torch.Tensor.dtype = torch.int8, scale=None):
        iinfo = torch.iinfo(int_dtype)
        if scale is None:
            scale = absmax_scale(base, int_dtype)
        elif scale.ndim > 0:
            if torch.squeeze(scale).ndim > 1:
                raise ValueError("Quantizing along multiple axis is not supported")
            if scale.ndim != base.ndim:
                raise ValueError(
                    "When quantizing per-axis, the scale must be broadcastable to the base (Tip: try to add missing dims of length zero)."
                )
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
        if t._data.dtype == torch.int32:
            # The dequantization operation might actually overflow in float16/bfloat16
            return (t._scale.to(torch.float32) * t._data).to(t._scale.dtype)
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


class QTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, scale, requires_grad=False):
        assert data.device == scale.device
        # This constructor can ONLY create leaf Tensors wrt autograd.
        # Use QTensor.from_tensor(t) to get a non-leaf Tensor wrt autograd.
        return torch.Tensor._make_wrapper_subclass(
            cls, data.size(), strides=data.stride(), dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, data, scale, requires_grad=False):
        self._axis = None
        if scale.ndim > 0:
            if torch.squeeze(scale).ndim > 1:
                raise ValueError("QTensor cannot be quantized along multiple axis.")
            if scale.ndim != data.ndim:
                raise ValueError(
                    "The QTensor scale must be broadcastable to the base (Tip: try to add missing dims of length zero)."
                )
            dims = scale.shape
            for i in range(scale.ndim):
                if dims[i] != 1:
                    self._axis = i
            if self._axis is None:
                # All dims are 1: the scale is actually a scalar
                scale = torch.squeeze(scale)
        self._data = data
        self._scale = scale

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

    @property
    def axis(self):
        return self._axis

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        from .func import get_qtensor_func

        kwargs = kwargs or {}

        # Look for a func accepting QTensor inputs
        qfunc = get_qtensor_func(func)
        if qfunc is not None:
            return qfunc(*args, **kwargs)
        # Defer to dispatcher to look instead for QTensor operations
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        from .ops import get_qtensor_op

        # Do not use directly func, but rather its overload
        op = op.overloadpacket
        # Look for a dispatched op accepting QTensor inputs
        qop = get_qtensor_op(op)
        if qop is not None:
            return qop(*args, **kwargs)
        # Identify the types of the args
        types = [type(arg).__name__ for arg in args]
        raise ValueError(f"{op} {types} is no supported for QTensor.")

    def numpy(self):
        return self.dequantize().cpu().numpy()
