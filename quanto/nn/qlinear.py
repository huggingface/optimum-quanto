from typing import Optional

import torch

from ..tensor import QBitsTensor, QTensor, absmax_scale, qint2, qint4, qint8, qtype
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLinear"]


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    def __init__(self, *args, weights: qtype = qint8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    @classmethod
    def from_module(cls, module, weights=qint8, activations: Optional[qtype] = None):
        qmodule = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            weights=weights,
            activations=activations,
            device=module.weight.device,
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
        if self.weights == qint8:
            wscale = absmax_scale(self.weight, axis=0)
            return QTensor.quantize(self.weight, qtype=self.weights, scale=wscale)
        elif self.weights in (qint2, qint4):
            return QBitsTensor.quantize(self.weight, qtype=self.weights, axis=0)
        raise ValueError(f"Invalid quantized weights type {self.weights}")

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activations is not None and not isinstance(input, QTensor):
            # Quantize tensor to be able to take advantage of accelerated matmul
            input = QTensor.quantize(input, self.activations, self.input_scale)
        # We always use quantized weights
        qweight = self.qweight()
        return torch.nn.functional.linear(input, qweight, bias=self.bias)
