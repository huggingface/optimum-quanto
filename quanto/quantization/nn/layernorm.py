from typing import Union

import torch

from ..qtensor import QTensor


class QLayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, list, torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.register_buffer("in_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("out_scale", torch.ones((), dtype=torch.float32))

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

    def freeze(self):
        # The weights of the LayerNorm are not quantized
        pass
