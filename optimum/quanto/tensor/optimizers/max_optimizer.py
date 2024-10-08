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

from typing import Tuple, Union

import torch

from ..qtype import qtype
from .affine_optimizer import AffineOptimizer


__all__ = ["MaxOptimizer"]


class MaxOptimizer(AffineOptimizer):
    def optimize(
        self, base: torch.Tensor, qtype: qtype, axis: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dim = list(range(1, base.ndim)) if (axis == 0) else list(range(0, base.ndim - 1))
        rmin = torch.amin(base, dim=dim, keepdim=True)
        rmax = torch.amax(base, dim=dim, keepdim=True)
        qmin = -(2 ** (qtype.bits - 1))
        qmax = 2 ** (qtype.bits - 1) - 1
        scale = (rmax - rmin) / (qmax - qmin)
        shift = -rmin
        return scale, shift
