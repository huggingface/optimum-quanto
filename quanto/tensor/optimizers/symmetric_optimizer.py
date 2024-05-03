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

from .optimizer import Optimizer


__all__ = ["SymmetricOptimizer"]


class SymmetricOptimizer(Optimizer):

    def __call__(self, base: torch.Tensor, bits: int, axis: Optional[int] = None) -> torch.Tensor:
        if axis not in [None, 0, -1]:
            raise ValueError("axis parameter must be None, 0 (first axis) or -1 (last axis)")
        scale = self.optimize(base, bits, axis)
        assert scale.dtype == base.dtype
        return scale

    def optimize(self, base: torch.Tensor, bits: int, axis: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError
