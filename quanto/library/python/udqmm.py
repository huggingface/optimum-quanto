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
    unpacked_shape: torch.Size,
):
    unpacked_weights = torch.ops.quanto.unpack(weights, bits)
    # TODO : we should proably add that in unpack with with arg unpacked_shape.
    unpacked_weights_resized = unpacked_weights[: unpacked_shape[0]]
    shifted_weights = unpacked_weights_resized.to(torch.int8) - zeropoint.to(torch.int8)
    scaled_weights = shifted_weights.to(scale.dtype) * scale
    ungrouped_weights = ungroup(scaled_weights, axis, orig_shape)
    return torch.ops.aten.mm(input, ungrouped_weights)
