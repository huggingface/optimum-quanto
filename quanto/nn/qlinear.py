from typing import Optional

import torch

from ..tensor import Optimizer, QTensor, qtype, quantize_activation
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLinear"]


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(
        cls, module, weights: qtype, activations: Optional[qtype] = None, optimizer: Optional[Optimizer] = None
    ):
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activation_qtype is not None and not isinstance(input, QTensor):
            # Quantize activations to be able to take advantage of accelerated matmul
            input = quantize_activation(input, qtype=self.activation_qtype, scale=self.input_scale)
        # We always use quantized weights
        return torch.nn.functional.linear(input, self.qweight, bias=self.bias)
