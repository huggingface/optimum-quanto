import torch


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
) -> torch.Tensor:
    # we transpose it back, so it is simpler to unpack since we have the pack + transposed weights
    weights = weights.transpose(0, 1)
    unpacked_weights = torch.ops.quanto.unpack(weights, bits)
    # TODO : we should proably add that in unpack with with arg unpacked_shape.
    # Depends if the weights have been transposed or not
    # if not transposed, we need to do unpacked_weights[: unpacked_shape[0]]
    unpacked_weights_resized = unpacked_weights[: unpacked_shape[1]]
    # transpose back
    unpacked_weights_resized = unpacked_weights_resized.transpose(0, 1)
    shifted_weights = unpacked_weights.to(torch.int8) - zeropoint
    scaled_weights = shifted_weights.to(scale.dtype) * scale
    ungrouped_weights = torch.ops.quanto.ungroup(scaled_weights, axis, orig_shape)
    return torch.ops.aten.mm(input, ungrouped_weights)
