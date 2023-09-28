import torch

from ..tensor import QuantizedTensor


class DynamicQLinear(torch.nn.Linear):
    """
    Activations are dynamically quantized for compute.
    Weights are statically quantized.
    Biases are not quantized.
    """

    @classmethod
    def from_module(cls, module):
        qmodule = cls(module.in_features, module.out_features, module.bias is not None)
        qmodule.weight = torch.nn.Parameter(QuantizedTensor.quantize(module.weight).to(module.weight.device))
        if module.bias is not None:
            qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    def forward(self, x):
        q_x = QuantizedTensor.quantize(x)
        return super().forward(self, q_x)


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
        # We only quantize the weight as the bias can only be quantized once we know the activation scale
        qmodule.weight = torch.nn.Parameter(QuantizedTensor.quantize(module.weight).to(module.weight.device))
        return qmodule.to(module.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QuantizedTensor):
            if input._data.dtype == torch.int32:
                # Reduce input bitwidth
                input = input.rescale(self.in_scale)
            out_int32 = torch.nn.functional.linear(input, self.weight, self.bias)
            return out_int32.rescale(self.out_scale)
        else:
            bias = None if self.bias is None else self.bias.dequantize()
            return torch.nn.functional.linear(input, self.weight.dequantize(), bias)
