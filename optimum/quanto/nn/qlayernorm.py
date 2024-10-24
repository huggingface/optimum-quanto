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


__all__ = ["QLayerNorm"]


@register_qmodule(torch.nn.LayerNorm)
class QLayerNorm(QModuleMixin, torch.nn.LayerNorm):
    @classmethod
    def qcreate(
        cls,
        module,
        weights: Optional[qtype] = None,
        activations: Optional[qtype] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        if activations is None:
            return None
        dtype = None if module.weight is None else module.weight.dtype
        return cls(
            module.normalized_shape,
            module.eps,
            module.elementwise_affine,
            module.bias is not None,
            dtype=dtype,
            device=device,
            weights=None,  # We never quantize QLayerNorm weights
            activations=activations,
            optimizer=None,  # We never quantize QLayerNorm weights
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
