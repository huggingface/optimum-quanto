import torch


@torch.library.impl("quanto_py::dqmm", "default")
def dqmm(input: torch.Tensor, other: torch.Tensor, other_scale: torch.Tensor):
    if other.dtype.is_floating_point:
        # An explicit type promotion avoids errors with float8 tensors (but is slower with integer)
        pother = other.to(other_scale.dtype)
        return torch.ops.aten.mm(input, pother * other_scale)
    return torch.ops.aten.mm(input, other * other_scale)
