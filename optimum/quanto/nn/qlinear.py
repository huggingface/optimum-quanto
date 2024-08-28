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


__all__ = ["QLinear"]


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
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
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=device,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.qweight, bias=self.bias)
