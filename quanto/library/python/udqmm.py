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
    packed_axis: int,
) -> torch.Tensor:
    unpacked_weights = torch.ops.quanto.unpack(weights, bits, unpacked_shape, packed_axis)
    shifted_weights = unpacked_weights.to(torch.int8) - zeropoint
    scaled_weights = shifted_weights.to(scale.dtype) * scale
    ungrouped_weights = torch.ops.quanto.ungroup(scaled_weights, axis, orig_shape)
    return torch.ops.aten.mm(input, ungrouped_weights)
