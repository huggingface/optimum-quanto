from typing import Union

import torch

from ..tensor import dtype_info


@torch.library.custom_op("quanto::quantize_symmetric", mutates_args=())
def quantize_symmetric(
    base: torch.Tensor, dtype: torch.dtype, axis: Union[int, None], scale: torch.Tensor
) -> torch.Tensor:
    # Sanity checks
    if axis is None:
        if scale.ndim > 0:
            raise ValueError("Scale must be a scalar when quantizing per-tensor")
    else:
        if base.ndim == 1:
            raise ValueError("1D Tensors cannot be quantized per-axis")
        if axis == base.ndim - 1:
            # Align on the general convention to index the last dimension
            axis = -1
        if axis not in (0, -1):
            raise ValueError("Quantization is only supported along the first or last axis.")
        if base.shape[axis] == 1:
            raise ValueError(f"Cannot quantize Tensor of shape {base.shape} along axis {axis} of size 1")
        if torch.squeeze(scale).ndim > 1:
            raise ValueError("Quantizing along multiple axis is not supported")
        if scale.ndim != base.ndim:
            raise ValueError(
                "When quantizing per-axis, the scale must be broadcastable to the base (Tip: try to add missing dims of length zero)."
            )
    data = base / scale
    if not dtype.is_floating_point:
        data = torch.round(data)
    info = dtype_info(dtype)
    return torch.clamp(data, min=info.min, max=info.max).to(dtype)
