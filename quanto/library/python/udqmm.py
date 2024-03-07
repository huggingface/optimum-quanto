import torch

from quanto.tensor.core import ungroup


@torch.library.impl("quanto_py::udqmm", "default")
def udqmm(
    input: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
    zeropoint: torch.Tensor,
    axis: int,
    bits: int,
    orig_shape: torch.Size,
):
    unpacked_weights = torch.ops.quanto.unpack(weights, bits)
    shifted_weights = unpacked_weights.to(torch.int8) - zeropoint.to(torch.int8)
    scaled_weights = shifted_weights * scale
    ungrouped_weights = ungroup(scaled_weights, axis, orig_shape)
    return torch.ops.aten.mm(input, ungrouped_weights)
