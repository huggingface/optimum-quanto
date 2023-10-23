import torch
from torch.autograd import Function


__all__ = ["QTensor"]


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

    @property
    def device(self):
        return self._data.device

    def numpy(self):
        return self.dequantize().cpu().numpy()
