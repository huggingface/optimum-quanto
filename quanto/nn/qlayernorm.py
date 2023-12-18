from typing import Optional

import torch

from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLayerNorm"]


@register_qmodule(torch.nn.LayerNorm)
class QLayerNorm(QModuleMixin, torch.nn.LayerNorm):
    @classmethod
    def from_module(cls, module, activations: Optional[torch.dtype] = None):
        if activations is None:
            return None
        qmodule = cls(
            module.normalized_shape,
            module.eps,
            module.elementwise_affine,
            module.bias is not None,
            activations=activations,
        )
        with torch.no_grad():
            qmodule.weight = torch.nn.Parameter(module.weight.clone().detach())
            if module.bias is not None:
                qmodule.bias = torch.nn.Parameter(module.bias.clone().detach())
        return qmodule.to(module.weight.device)

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
