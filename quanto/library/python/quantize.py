import torch


def dtype_info(dtype):
    info = torch.finfo if dtype.is_floating_point else torch.iinfo
    return info(dtype)


@torch.library.impl("quanto_py::quantize_symmetric", "default")
def quantize_symmetric(t: torch.Tensor, scale: torch.Tensor, dtype: torch.Tensor.dtype):
    info = dtype_info(dtype)
    data = t / scale
    if not dtype.is_floating_point:
        data = torch.round(data)
    return torch.clamp(data, min=info.min, max=info.max).to(dtype)
