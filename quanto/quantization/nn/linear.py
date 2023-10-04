import torch

from ..tensor import QuantizedTensor


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
        return qmodule.to(module.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QuantizedTensor):
            if input._data.dtype == torch.int32:
                # Reduce input bitwidth
                input = input.rescale(torch.int8, self.in_scale)
            # If needed, quantize weights and bias
            weight = self.weight
            if not isinstance(weight, QuantizedTensor):
                weight = QuantizedTensor.quantize(weight)
            bias = self.bias
            if bias is not None:
                bias_scale = self.in_scale * weight._scale
                if isinstance(bias, QuantizedTensor):
                    if bias._scale != bias_scale:
                        # This should only happen if we calibrate again a frozen module
                        bias = QuantizedTensor.rescale(torch.int32, bias_scale)
                else:
                    bias = QuantizedTensor.quantize(bias, torch.int32, bias_scale)
            # Operate on quantized tensor
            out_int32 = torch.nn.functional.linear(input, weight, bias)
            # Downscale
            return out_int32.rescale(torch.int8, self.out_scale)
        else:
            bias = None if self.bias is None else self.bias.dequantize()
            return torch.nn.functional.linear(input, self.weight.dequantize(), bias)

    def freeze(self):
        # Replace float weights by quantized weights
        self.weight = torch.nn.Parameter(QuantizedTensor.quantize(self.weight).to(self.weight.device))
        if self.bias is not None:
            bias_scale = self.in_scale * self.weight._scale
            self.bias = torch.nn.Parameter(QuantizedTensor.quantize(self.bias, torch.int32, bias_scale))
