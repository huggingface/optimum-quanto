import torch

from .qtype import qtype
from .quantizers import SymmetricQuantizer


__all__ = ["quantize_activation"]


def quantize_activation(t: torch.Tensor, qtype: qtype, scale: torch.Tensor):
    """Quantize an activation Tensor.

    Activations are always quantized per-tensor with a scalar scale.

    Args:
        base (`torch.Tensor`): the Tensor to quantize
        qtype (`quanto.qtype`): The target quantization type
        scale (`torch.Tensor`): The scalar quantization scale

    Returns:
        A quantized Tensor.
    """
    if scale.numel() != 1:
        raise ValueError("Parameter scale must be a scalar because activations can only be quantized per-tensor")
    return SymmetricQuantizer.apply(t, qtype, None, None, scale)
