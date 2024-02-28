from typing import Optional

import torch

from ..tensor import QTensor, qtype
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QConv2d"]


@register_qmodule(torch.nn.Conv2d)
class QConv2d(QModuleMixin, torch.nn.Conv2d):
    @classmethod
    def qcreate(cls, module, weights: qtype, activations: Optional[qtype] = None):
        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            dtype=module.weight.dtype,
            device=module.weight.device,
            weights=weights,
            activations=activations,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activations is not None and not isinstance(input, QTensor):
            # Quantize tensor to be able to take advantage of accelerated conv2d
            input = QTensor.quantize(input, qtype=self.activations, axis=None, scale=self.input_scale)
        # We always use quantized weights
        qweight = self.qweight()
        return self._conv_forward(input, qweight, self.bias)
