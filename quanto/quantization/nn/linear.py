import torch

from ..qtensor import QTensor


class QLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, dtype, device)
        self.register_buffer("in_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("out_scale", torch.ones((), dtype=torch.float32))

    @classmethod
    def from_module(cls, module):
        qmodule = cls(module.in_features, module.out_features, module.bias is not None)
        with torch.no_grad():
            qmodule.weight.copy_(module.weight)
            if module.bias is not None:
                qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If needed, quantize inputs, weights and bias
        if isinstance(input, QTensor):
            if input._data.dtype == torch.int32:
                # Reduce input bitwidth
                input = input.rescale(torch.int8, self.in_scale)
        else:
            input = QTensor.quantize(input, torch.int8, self.in_scale)
        weight = self.weight
        if not isinstance(weight, QTensor):
            weight = QTensor.quantize(weight)
        bias = self.bias
        if bias is not None:
            bias_scale = self.in_scale * weight._scale
            if isinstance(bias, QTensor):
                if bias._scale != bias_scale:
                    # This should only happen if we calibrate again a frozen module
                    bias = QTensor.rescale(torch.int32, bias_scale)
            else:
                bias = QTensor.quantize(bias, torch.int32, bias_scale)
        # Operate on quantized tensors
        out_int32 = torch.nn.functional.linear(input, weight, bias)
        # Downscale
        return out_int32.rescale(torch.int8, self.out_scale)

    def freeze(self):
        # Replace float weights by quantized weights
        self.weight = torch.nn.Parameter(QTensor.quantize(self.weight).to(self.weight.device))
        if self.bias is not None:
            bias_scale = self.in_scale * self.weight._scale
            self.bias = torch.nn.Parameter(QTensor.quantize(self.bias, torch.int32, bias_scale))
