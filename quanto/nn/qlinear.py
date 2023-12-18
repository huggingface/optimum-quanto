from typing import Optional

import torch

from ..qtensor import QTensor, absmax_scale
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLinear"]


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    def __init__(self, *args, weights: torch.dtype = torch.int8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    @classmethod
    def from_module(cls, module, weights=torch.int8, activations: Optional[torch.dtype] = None):
        qmodule = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            weights=weights,
            activations=activations,
        )
        with torch.no_grad():
            qmodule.weight.copy_(module.weight)
            if module.bias is not None:
                qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    def qweight(self):
        if isinstance(self.weight, QTensor):
            return self.weight
        # Quantize the weights per-axis
        wscale = absmax_scale(self.weight, axis=0)
        return QTensor.quantize(self.weight, itype=self.weights, scale=wscale)

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activations is not None and not isinstance(input, QTensor):
            # Quantize tensor to be able to take advantage of accelerated matmul
            input = QTensor.quantize(input, self.activations, self.input_scale)
        # We always use quantized weights
        qweight = self.qweight()
        output = torch.matmul(input, qweight.t())
        if self.bias is not None:
            # The outputs will be dequantized in the addition since the biases are not quantized
            output = output + self.bias
        return output
