import math
from typing import List

import torch


__all__ = ["group", "ungroup", "grouped_shape"]


def grouped_shape(shape: List, axis: int, group_size: int) -> List:
    if axis not in (0, -1):
        raise ValueError("Axis must be 0 or -1 for group-wise quantization")
    n_groups = math.prod(shape) // group_size
    return (n_groups, group_size) if axis == 0 else (group_size, n_groups)


def group(base: torch.Tensor, axis: int, group_size: int):
    if axis not in (0, -1):
        raise ValueError("Axis must be 0 or -1 for group-wise quantization")
    # In standard per-axis quantization, we have one scale per axis dim
    axis_dim = base.shape[axis]
    # This scale is evaluated over axis_numel items for each feature along axis
    axis_numel = base.numel() // axis_dim
    if group_size > axis_numel or axis_numel % group_size != 0:
        raise ValueError(f"Group size ({group_size}) must be a divisor of ({axis_numel})")
    # Group-wise quantization further splits axis_numel into multiple groups per axis
    axis_groups = axis_numel // group_size
    if axis == 0:
        # Easy-peasy: we simply need to reshape to (axis_dim * axis_groups, group_size)
        return base.reshape([-1, group_size])
    # More difficult: reshape to (group_size, axis_dim * axis_groups)
    # First, split by groups, preserving the axis dimension
    grouped = base.reshape((axis_groups, group_size, axis_dim))
    # Permute to (group_size, axis_dim, axis_groups)
    grouped = grouped.permute(1, 2, 0)
    return grouped.reshape(group_size, axis_dim * axis_groups)


def ungroup(grouped: torch.Tensor, axis: int, orig_shape: torch.Size):
    if grouped.shape == orig_shape:
        return grouped
    if axis == 0:
        # No transposition required, just reshape
        return grouped.reshape(orig_shape)
    group_size = grouped.shape[0] if axis == -1 else grouped.shape[-1]
    axis_dim = orig_shape[axis]
    axis_groups = grouped.numel() // axis_dim // group_size
    ungrouped = grouped.reshape(group_size, axis_dim, axis_groups)
    # Permute to (axis_groups, group_size, axis_dim)
    ungrouped = ungrouped.permute(2, 0, 1)
    return ungrouped.reshape(orig_shape)
