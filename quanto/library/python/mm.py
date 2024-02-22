import torch


@torch.library.impl("quanto_py::dqmm", "default")
def dqmm(input: torch.Tensor, other: torch.Tensor, other_scale: torch.Tensor):
    return torch.ops.aten.mm(input, other * other_scale)
