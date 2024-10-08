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

from typing import Optional, Tuple, Union

import torch

from ..qtype import qtype
from .symmetric_optimizer import SymmetricOptimizer


__all__ = ["AbsmaxOptimizer"]


class AbsmaxOptimizer(SymmetricOptimizer):
    def optimize(
        self, base: torch.Tensor, qtype: qtype, axis: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        base = torch.abs(base)
        if axis is None:
            rmax = torch.max(base)
        else:
            dim = list(range(1, base.ndim)) if (axis == 0) else list(range(0, base.ndim - 1))
            rmax = torch.amax(torch.abs(base), dim=dim, keepdim=True)
        return rmax / qtype.qmax
