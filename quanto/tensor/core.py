from typing import Optional

import torch

from .qtype import qint8, qtype


__all__ = ["absmax_scale", "axis_to_dim", "dtype_info"]


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
