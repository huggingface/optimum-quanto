import torch
from torch.autograd import Function

from ..core import group
from ..packed import PackedTensor
from ..qbits import QBitsTensor
from ..qtype import qint2, qint4, qtype


__all__ = ["AffineQuantizer"]


class AffineQuantizer(Function):
    """A standard affine quantizer."""

    @staticmethod
    def forward(
        ctx, base: torch.Tensor, qtype: qtype, axis: int, group_size: int, scale: torch.Tensor, zeropoint: torch.Tensor
    ):
        if qtype not in (qint2, qint4):
            raise ValueError("QBitsTensor can only be of qint2 or qint4 qtype")
        if axis not in (0, -1):
            raise ValueError("QBitsTensor axis parameter must be 0 (first axis) or -1 (last axis)")
        size = base.size()
        stride = base.stride()
        if group_size is not None:
            base = group(base, axis=axis, group_size=group_size)
        bits = qtype.bits
        data = torch.clamp(torch.round(base / scale) + zeropoint, min=0, max=2**bits - 1).to(torch.uint8)
        packed = PackedTensor.pack(data, bits)

        return QBitsTensor(qtype, axis, size, stride, packed, scale, zeropoint)

    @staticmethod
    def backward(ctx, gO):
        # For autograd, quantization is a no-op
        return gO, None, None, None, None, None
