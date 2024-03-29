from typing import Optional

import torch

from .optimizers import AbsmaxOptimizer, AffineOptimizer, MaxOptimizer, Optimizer, SymmetricOptimizer
from .qtype import qtype
from .quantizers import AffineQuantizer, SymmetricQuantizer


__all__ = ["quantize_weight"]


default_affine_optimizer = MaxOptimizer()
default_symmetric_optimizer = AbsmaxOptimizer()


def quantize_weight(
    t: torch.Tensor, qtype: qtype, axis: int, group_size: Optional[int] = None, optimizer: Optional[Optimizer] = None
):
    """Quantize a weight Tensor.

    Weights are always quantized per-axis.

    Args:
        t (`torch.Tensor`): the weight Tensor to quantize
        qtype (`quanto.qtype`): The target quantization type
        axis ('int`): The quantization axis (0 or -1)
        group_size (`Optional[int]`): The quantization group size
        optimizer (`Optional[quanto.Optimizer]`): An optimizer to evaluate the scale if not provided.
            Defaults to a max Optimizer.

    Returns:
        A quantized Tensor.
    """
    if axis not in (0, -1):
        raise ValueError("axis parameter must be 0 (first axis) or -1 (last axis)")
    if qtype.bits == 8:
        if optimizer is None:
            optimizer = default_symmetric_optimizer
        else:
            if not isinstance(optimizer, SymmetricOptimizer):
                raise ValueError("A SymmetricOptimizer is expected")
        scale = optimizer(t, qtype.bits, axis, group_size)
        return SymmetricQuantizer.apply(t, qtype, axis, group_size, scale)
    if optimizer is None:
        optimizer = default_affine_optimizer
    else:
        if not isinstance(optimizer, AffineOptimizer):
            raise ValueError("An AffineOptimizer is expected")
    scale, zeropoint = optimizer(t, qtype.bits, axis, group_size)
    return AffineQuantizer.apply(t, qtype, axis, group_size, scale, zeropoint)
