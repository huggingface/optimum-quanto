import torch

from ..qtensor import QTensor, absmax_scale
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLinear"]


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    @classmethod
    def from_module(cls, module):
        qmodule = cls(module.in_features, module.out_features, module.bias is not None, dtype=module.weight.dtype)
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
        return QTensor.quantize(self.weight, scale=wscale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If needed, quantize inputs
        if self.scales.input is not None:
            if isinstance(input, QTensor):
                if input.itype == torch.int32:
                    # Requantize input to per-tensor int8
                    input = input.rescale(torch.int8, self.scales.input)
            else:
                input = QTensor.quantize(input, torch.int8, self.scales.input)
        # We always use quantized weights
        qweight = self.qweight()
        # The weights might be dequantized in the matmul if the inputs are not quantized
        output = torch.matmul(input, qweight.t())
        if self.bias is not None:
            # The outputs will be dequantized in the addition since the biases are not quantized
            output = output + self.bias
        if self.scales.output is not None:
            if isinstance(output, QTensor):
                # Downscale to int8
                output = output.rescale(torch.int8, self.scales.output)
            else:
                output = QTensor.quantize(output, torch.int8, self.scales.output)
        return output
