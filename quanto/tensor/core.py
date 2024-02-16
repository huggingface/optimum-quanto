from typing import Optional

import torch
from torch.autograd import Function
from torch.utils import _pytree as pytree

from .qtype import qint2, qint4, qint8, qtype


__all__ = ["absmax_scale", "qfallback", "dtype_info", "QBitsTensor", "QTensor"]


def dtype_info(dtype):
    info = torch.finfo if dtype.is_floating_point else torch.iinfo
    return info(dtype)


def axis_to_dim(t, axis):
    dim = list(range(t.ndim))
    if axis == -1:
        dim = dim[:-1]
    else:
        dim.remove(axis)
    return dim


def pack_weights(intweights: torch.Tensor, qtype: qtype) -> torch.Tensor:
    """
    Pack int4 / int2 weights in a unint8 tensor

    What packing means? Assume we have 4 values that are in 2bit but encoded in 8bit
    (because torch does not have native support for 2-bit datatypes)

    > 0000 0011 | 0000 0010 | 0000 0001 | 0000 0000

    We can pack them in a single 8-bit uint value

    > 1110 0100

    Therefore instead of saving 4 values in 8-bit precision we save a single value of 8-bit precision saving 24 bits in total.

    Args:
        intweights (`torch.Tensor`):
            The un-packed tensor in `torch.int8` precision
        qtype (`quanto.qtype`):
            The desired `qtype` - can be `quanto.int2`, `quanto.int4`
    """
    bits = qtype.bits
    original_shape = intweights.shape
    values_per_item = 8 // bits
    if original_shape[0] % values_per_item != 0:
        # We cannot pack: return the original tensor
        return intweights
    row_dim = original_shape[0] // values_per_item

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    packed = torch.zeros(packed_tensor_shape, device=intweights.device, dtype=torch.uint8)
    unpacked = intweights.to(torch.uint8)

    def lshift(t: torch.Tensor, bits: int):
        if t.device.type == "mps":
            # lshift is not supported on MPS device
            return t * (2**bits)
        return t << bits

    for i in range(values_per_item):
        packed |= lshift(unpacked[i * row_dim : (i + 1) * row_dim], bits * i)

    return packed


def unpack_weights(uint8weights: torch.Tensor, qtype: qtype) -> torch.Tensor:
    """
    Un-Pack int4 / int2 weights (packed in a uint8) into a torch.int8 tensor
    What un-packing means? Assume we have packed 4 2-bit values in 8-bit
    (because torch does not have native support for 2-bit datatypes)

    > 1110 0100

    Unpacking them means retrieving the original 4 2-bit values:

    > 0000 0011 | 0000 0010 | 0000 0001 | 0000 0000

    Args:
        uint8weights (`torch.Tensor`):
            The packed tensor in `torch.uint8` precision
        qtype (`quanto.qbitsdtype`):
            The desired `qtype` - can be `quanto.int2`, `quanto.int4`
    """
    bits = qtype.bits
    unpacked = []
    values_per_item = 8 // bits

    def rshift(t: torch.Tensor, bits: int):
        if t.device.type == "mps":
            # rshift is not supported on MPS device
            return t // (2**bits)
        return t >> bits

    # Unpack each set of values independently
    for i in range(values_per_item):
        mask = 2 ** (bits * (i + 1)) - 1
        unpacked.append(rshift(uint8weights & mask, bits * i))
    # Return the concatenated unpacked tensors
    return torch.cat(unpacked).to(torch.int8)


def absmax_scale(base: torch.Tensor, qtype: qtype = qint8, axis: Optional[int] = None) -> torch.Tensor:
    """Evaluate the quantization scale using the absmax algorithm.

    The Absolute Maximum quantization algorithm is a symmetrical quantization
    algorithm where the scale corresponds to the maximum absolute value of the
    base divided by the highest positive integer value for the target integer
    representation.

    Args:
        base (`torch.Tensor`): the base tensor on which the scale will be applied.
        qtype (`quanto.qtype`): the target qtype for quantization.
        axis (`int`): the index of the axis to preserve, or -1 for the last one.
            Defaults to None to reduce all axis.

    Returns:
        `torch.Tensor`: a scale tensor of the same dtype as the base.
    """
    abs_base = torch.abs(base)
    if axis is None:
        qranges = torch.max(abs_base)
    else:
        dim = axis_to_dim(abs_base, axis)
        qranges = torch.amax(torch.abs(base), dim=dim, keepdim=True)
    info = dtype_info(qtype.dtype)
    return qranges / info.max


def qfallback(callable, *args, **kwargs):
    """Fallback method for QTensor inputs.

    When a torch function or an aten operation is not supported for the specified
    QTensor arguments, each QTensor arg or kwarg is dequantized to a torch.Tensor
    before calling the target function or op.
    """
    args, kwargs = pytree.tree_map_only(QTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


class Quantizer(Function):
    """A standard symmetric quantizer.

    If the quantization scale is not specified, it uses the absmax scale for the base tensor.
    """

    @staticmethod
    def forward(ctx, base, qtype: qtype = qint8, scale=None):
        info = dtype_info(qtype.dtype)
        if scale is None:
            scale = absmax_scale(base, qtype)
        elif scale.ndim > 0:
            if torch.squeeze(scale).ndim > 1:
                raise ValueError("Quantizing along multiple axis is not supported")
            if scale.ndim != base.ndim:
                raise ValueError(
                    "When quantizing per-axis, the scale must be broadcastable to the base (Tip: try to add missing dims of length zero)."
                )
        data = base / scale
        if not qtype.is_floating_point:
            data = torch.round(data)
        data = torch.clamp(data, min=info.min, max=info.max).to(qtype.dtype)
        # The instantiation of the quantized tensor must happen within the context of the Function
        # for the autograd magic to work.
        return QTensor(qtype, data, scale)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None


class Dequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        if t.qtype == torch.int32:
            # The dequantization operation requires data to be cast to the scale float type before multiplication
            # by the scale, but this might actually overflow for float16/bfloat16
            return (t._scale.to(torch.float32) * t._data).to(t._scale.dtype)
        elif t.qtype.is_floating_point:
            # Upcast explicitly to the scale dtype
            return t._scale * t._data.to(t._scale.dtype)
        return t._scale * t._data

    @staticmethod
    def backward(ctx, gO):
        # For autograd, dequantization is a no-op
        return gO


class QTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, qtype, data, scale, requires_grad=False):
        assert data.device == scale.device
        # This constructor can ONLY create leaf Tensors wrt autograd.
        # Use QTensor.from_tensor(t) to get a non-leaf Tensor wrt autograd.
        return torch.Tensor._make_wrapper_subclass(
            cls, data.size(), strides=data.stride(), dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, data, scale, requires_grad=False):
        self._qtype = qtype
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
            elif self._axis == scale.ndim - 1:
                # Align on the general convention to index the last dimension
                self._axis = -1
        self._data = data
        self._scale = scale

    def __repr__(self):  # Zero out missing values for printing
        autograd_info = (
            f", grad_fn={self.grad_fn}" if self.grad_fn else ", requires_grad=True" if self.requires_grad else ""
        )
        return f"QTensor({self._data}, scale={self._scale}, public_dtype={self.dtype}{autograd_info})"

    @classmethod
    def quantize(cls, base, qtype=qint8, scale=None):
        """Differentiable quantization function"""
        return Quantizer.apply(base, qtype, scale)

    def dequantize(self):
        """Differentiable dequantization function"""
        return Dequantizer.apply(self)

    @property
    def axis(self):
        return self._axis

    @property
    def qtype(self):
        return self._qtype

    def __tensor_flatten__(self):
        return ["_data", "_scale"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        assert meta is None
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        return QTensor(data, scale)

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
        from .ops import get_qtensor_op_dispatch

        # Do not use directly func, but rather its overload
        op = op.overloadpacket
        # Look for a dispatched op accepting QTensor inputs
        qdispatch = get_qtensor_op_dispatch(op)
        if qdispatch is not None:
            if qdispatch.qargs is not None:
                # Check QTensor arguments
                for qarg in qdispatch.qargs:
                    arg = args[qarg.index]
                    if not isinstance(arg, QTensor) or arg.axis not in qarg.axis:
                        # Incompatible argument detected: qfallback
                        return qfallback(op, *args, **kwargs)
            return qdispatch.qop(*args, **kwargs)
        # No dispatch available: qfallback
        return qfallback(op, *args, **kwargs)

    def numpy(self):
        return self.dequantize().cpu().numpy()


class AffineQuantizer(Function):
    """A standard affine quantizer."""

    @staticmethod
    def forward(ctx, base, qtype: qtype, axis=None, pack=False):
        assert qtype in (qint2, qint4)
        bits = qtype.bits
        if axis is None:
            rmin = torch.min(base)
            rmax = torch.max(base)
        else:
            dim = axis_to_dim(base, axis)
            rmin = torch.amin(base, dim=dim, keepdim=True)
            rmax = torch.amax(base, dim=dim, keepdim=True)
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = (rmax - rmin) / (qmax - qmin)
        zeropoint = torch.round(-rmin / scale).to(torch.int8)
        data = torch.clamp(torch.round((base - rmin) / scale), min=0, max=2**bits - 1).to(torch.int8)

        if pack and data.dtype == torch.int8:
            data = pack_weights(data, qtype)

        return QBitsTensor(qtype, data, scale, zeropoint)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None


class QBitsToQTensor(Function):
    @staticmethod
    def forward(ctx, t):
        if t.packed:
            unpacked_data = unpack_weights(t._data, t.qtype)
        else:
            unpacked_data = t._data
        int8_data = unpacked_data.to(torch.int8) - t._zeropoint.to(torch.int8)
        return QTensor(qint8, int8_data, t._scale)

    @staticmethod
    def backward(ctx, gO):
        return gO


class QBitsTensor(QTensor):
    @staticmethod
    def __new__(cls, qtype, data, scale, zeropoint, requires_grad=False):
        assert data.device == scale.device
        assert data.device == zeropoint.device
        packed = data.dtype == torch.uint8
        size = data.size()
        if packed:
            # Fixme: create a PackedIntTensor subclass to store the packed / shape info
            size = (size[0] * (8 // qtype.bits), *size[1:])
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=data.stride(), dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, data, scale, zeropoint, requires_grad=False):
        super().__init__(qtype, data, scale, requires_grad=requires_grad)
        self._zeropoint = zeropoint
        self.packed = data.dtype == torch.uint8

    def __repr__(self):
        autograd_info = (
            f", grad_fn={self.grad_fn}" if self.grad_fn else ", requires_grad=True" if self.requires_grad else ""
        )
        return f"QBitsTensor({self._data}, scale={self._scale}, zeropoint={self._zeropoint}, dtype={self.dtype}{autograd_info})"

    @classmethod
    def quantize(cls, base, qtype=qint4, axis=None, pack=True):
        """Differentiable quantization function"""
        return AffineQuantizer.apply(base, qtype, axis, pack)

    def qtensor(self):
        return QBitsToQTensor.apply(self)

    def dequantize(self):
        return self.qtensor().dequantize()

    @property
    def qtype(self):
        return self._qtype

    def __tensor_flatten__(self):
        return ["_data", "_scale", "_zeropoint"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 3
        assert meta is None
        data, scale, zeropoint = inner_tensors["_data"], inner_tensors["_scale"], inner_tensors["_zeropoint"]
        return QTensor(data, scale, zeropoint)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            scale = op(t._scale)
            zeropoint = op(t._zeropoint)
            return QBitsTensor(t._qtype, data, scale, zeropoint)
        args, kwargs = pytree.tree_map_only(QBitsTensor, lambda x: x.qtensor(), (args, kwargs or {}))
        return op(*args, **kwargs)
