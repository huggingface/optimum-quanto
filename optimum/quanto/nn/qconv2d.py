# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from ..tensor import Optimizer, qtype
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QConv2d"]


@register_qmodule(torch.nn.Conv2d)
class QConv2d(QModuleMixin, torch.nn.Conv2d):
    @classmethod
    def qcreate(
        cls,
        module,
        weights: qtype,
        activations: Optional[qtype] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
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
            device=device,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.qweight, self.bias)
