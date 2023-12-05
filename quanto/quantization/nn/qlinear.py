import torch

from ..qtensor import QTensor, absmax_scale
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLinear"]


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    @classmethod
    def from_module(cls, module):
        qmodule = cls(module.in_features, module.out_features, module.bias is not None)
        with torch.no_grad():
            qmodule.weight.copy_(module.weight)
            if module.bias is not None:
                qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    def qparams(self):
        qweight = self.weight
        if not isinstance(qweight, QTensor):
            # Quantize the weights per-axis if the outputs are per-axis
            axis = None if self.out_scale.ndim == 0 else 0
            wscale = absmax_scale(self.weight, axis=axis)
            qweight = QTensor.quantize(self.weight, scale=wscale)
        qbias = self.bias
        if qbias is not None:
            bias_scale = torch.squeeze(self.in_scale * qweight._scale)
            if isinstance(qbias, QTensor):
                if torch.any(qbias._scale != bias_scale):
                    # This should only happen if we calibrate again a frozen module
                    qbias = qbias.rescale(torch.int16, bias_scale)
            else:
                qbias = QTensor.quantize(qbias, torch.int16, bias_scale)
        return qweight, qbias

    def qweight(self):
        if isinstance(self.weight, QTensor):
            return self.weight
        # Quantize the weights per-axis if the outputs are per-axis
        axis = None if self.out_scale.ndim == 0 else 0
        wscale = absmax_scale(self.weight, axis=axis)
        return QTensor.quantize(self.weight, scale=wscale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If needed, quantize inputs
        if isinstance(input, QTensor):
            if input._data.dtype == torch.int32:
                # Requantize input to per-tensor int8
                input = input.rescale(torch.int8, self.in_scale)
        else:
            input = QTensor.quantize(input, torch.int8, self.in_scale)
        # Operate on quantized tensors
        qweight, qbias = self.qparams()
        output = torch.nn.functional.linear(input, qweight, qbias)
        if isinstance(output, QTensor):
            # Downscale
            output = output.rescale(torch.int8, self.out_scale)
        return output

    def freeze(self):
        # Replace float weights by quantized weights
        qweight, qbias = self.qparams()
        self.weight = torch.nn.Parameter(qweight)
        if self.bias is not None:
            self.bias = torch.nn.Parameter(qbias)
