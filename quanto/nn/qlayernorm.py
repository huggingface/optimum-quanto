from typing import Optional

import torch

from ..tensor import qtype
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLayerNorm"]


@register_qmodule(torch.nn.LayerNorm)
class QLayerNorm(QModuleMixin, torch.nn.LayerNorm):
    @classmethod
    def qcreate(cls, module, weights: Optional[qtype] = None, activations: Optional[qtype] = None):
        if activations is None:
            return None
        return cls(
            module.normalized_shape,
            module.eps,
            module.elementwise_affine,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
            weights=None,  # We never quantize QLayerNorm weights
            activations=activations,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
