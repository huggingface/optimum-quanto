from copy import copy

import torch
from torch.autograd import Function
from torch.utils import _pytree as pytree

from .core import axis_to_dim
from .packed import PackedTensor
from .qtensor import QTensor
from .qtype import qint2, qint4, qint8, qtype


__all__ = ["QBitsTensor"]


class AffineQuantizer(Function):
    """A standard affine quantizer."""

    @staticmethod
    def forward(ctx, base, qtype: qtype, axis=None):
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
        data = torch.clamp(torch.round((base - rmin) / scale), min=0, max=2**bits - 1).to(torch.uint8)
        packed = PackedTensor.pack(data, bits)

        return QBitsTensor(qtype, packed, scale, zeropoint)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None


class QBitsToQTensor(Function):
    @staticmethod
    def forward(ctx, t):
        unpacked = t._data.unpack()
        int8_data = unpacked.to(torch.int8) - t._zeropoint.to(torch.int8)
        return QTensor(qint8, int8_data, t._scale)

    @staticmethod
    def backward(ctx, gO):
        return gO


class QBitsTensor(QTensor):
    @staticmethod
    def __new__(cls, qtype, data, scale, zeropoint, requires_grad=False):
        assert isinstance(data, PackedTensor)
        assert data.device == scale.device
        assert data.device == zeropoint.device
        size = data.size()
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=data.stride(), dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, data, scale, zeropoint, requires_grad=False):
        super().__init__(qtype, data, scale, requires_grad=requires_grad)
        self._zeropoint = zeropoint

    def __repr__(self):
        autograd_info = (
            f", grad_fn={self.grad_fn}" if self.grad_fn else ", requires_grad=True" if self.requires_grad else ""
        )
        return f"QBitsTensor({self._data}, scale={self._scale}, zeropoint={self._zeropoint}, dtype={self.dtype}{autograd_info})"

    @classmethod
    def quantize(cls, base, qtype=qint4, axis=None):
        """Differentiable quantization function"""
        return AffineQuantizer.apply(base, qtype, axis)

    def qtensor(self):
        return QBitsToQTensor.apply(self)

    def dequantize(self):
        return self.qtensor().dequantize()

    @property
    def qtype(self):
        return self._qtype

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_zeropoint"]
        meta = {"qtype": self._qtype}
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 3
        assert len(meta) == 1
        data, scale, zeropoint = inner_tensors["_data"], inner_tensors["_scale"], inner_tensors["_zeropoint"]
        qtype = meta["qtype"]
        return QTensor(qtype, data, scale, zeropoint)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        if op.overloadpacket is torch.ops.aten.detach:
            # Detach is required when copying and deserializing
            t = args[0]
            data = op(t._data)
            scale = op(t._scale)
            zeropoint = op(t._zeropoint)
            return QBitsTensor(t._qtype, data, scale, zeropoint)
        elif op.overloadpacket is torch.ops.aten._to_copy:
            t = args[0]
            # Copy scale
            scale = op(t._scale, **kwargs)
            # Move data and zeropoint, ignoring dtype (it only applies to scale)
            data_kwargs = copy(kwargs)
            data_kwargs["dtype"] = torch.uint8
            data = op(t._data, **data_kwargs)
            zeropoint_kwargs = copy(kwargs)
            zeropoint_kwargs["dtype"] = torch.int8
            zeropoint = op(t._data, **data_kwargs)
            return QBitsTensor(t._qtype, data, scale, zeropoint)
        args, kwargs = pytree.tree_map_only(QBitsTensor, lambda x: x.qtensor(), (args, kwargs or {}))
        return op(*args, **kwargs)
