import torch

from ..qtensor import QTensor
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLayerNorm"]


@register_qmodule(torch.nn.LayerNorm)
class QLayerNorm(QModuleMixin, torch.nn.LayerNorm):
    @classmethod
    def from_module(cls, module):
        qmodule = cls(module.normalized_shape, module.eps, module.elementwise_affine, module.bias is not None)
        with torch.no_grad():
            qmodule.weight.copy_(module.weight)
            if module.bias is not None:
                qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If needed, dequantize inputs
        if isinstance(input, QTensor):
            input = input.dequantize()
        out = torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        # Quantize output
        return QTensor.quantize(out, torch.int8, self.out_scale)
