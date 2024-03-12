import torch


@torch.library.impl("quanto_py::ungroup", "default")
def ungroup(grouped: torch.Tensor, axis: int, orig_shape: torch.Size) -> torch.Tensor:
    if grouped.shape == orig_shape:
        return grouped
    if axis == 0:
        # No transposition required, just reshape
        return grouped.reshape(orig_shape)
    group_size = grouped.shape[0] if axis == -1 else grouped.shape[-1]
    axis_dim = orig_shape[axis]
    groups_per_axis = grouped.numel() // axis_dim // group_size
    ungrouped = grouped.reshape(group_size, axis_dim, groups_per_axis)
    # A dual tranposition is required to reorder to (groups_per_axis, group_size, axis_dim)
    ungrouped = ungrouped.transpose(1, 2)
    ungrouped = ungrouped.transpose(0, 1)
    return ungrouped.reshape(orig_shape)
