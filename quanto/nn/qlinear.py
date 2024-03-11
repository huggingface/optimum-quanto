from typing import Optional

import torch

from ..tensor import QTensor, qtype
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLinear"]


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(cls, module, weights: qtype, activations: Optional[qtype] = None):
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
            weights=weights,
            activations=activations,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activation_qtype is not None and not isinstance(input, QTensor):
            # Quantize activations to be able to take advantage of accelerated matmul
            input = QTensor.quantize(
                input, qtype=self.activation_qtype, axis=None, group_size=None, scale=self.input_scale
            )
        # We always use quantized weights
        qweight = self.qweight()
        return torch.nn.functional.linear(input, qweight, bias=self.bias)
