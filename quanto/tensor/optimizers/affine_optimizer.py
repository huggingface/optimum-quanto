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

from typing import Optional, Tuple

import torch

from ..qbits import group
from .optimizer import Optimizer


__all__ = ["AffineOptimizer"]


class AffineOptimizer(Optimizer):

    def __call__(
        self, base: torch.Tensor, bits: int, axis: int, group_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if axis not in [0, -1]:
            raise ValueError("axis parameter must be 0 (first axis) or -1 (last axis)")
        if group_size is not None:
            base = group(base, axis, group_size)
        scale, zeropoint = self.optimize(base, bits, axis)
        assert scale.dtype == base.dtype
        assert zeropoint.dtype == torch.int8
        return scale, zeropoint

    def optimize(self, base: torch.Tensor, bits: int, axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
