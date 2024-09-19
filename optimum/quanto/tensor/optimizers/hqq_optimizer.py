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
from ..weights import quantize_weight
from .max_optimizer import MaxOptimizer


__all__ = ["HqqOptimizer"]


# Shrinking operator
def shrink_lp_op(x: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )


class HqqOptimizer(MaxOptimizer):
    """Implementation of the HQQ algorithm

    This is an implementation of the algorithm described in "Half-Quadratic Quantization of Large Machine Learning Models",
    by Hicham Badri and Appu Shaji (https://mobiusml.github.io/hqq_blog/).
    This is an adaption of the original implementation at https://github.com/mobiusml/hqq.

    """

    def __init__(
        self,
        lp_norm: Optional[float] = 0.7,
        beta: Optional[int] = 1e1,
        kappa: Optional[float] = 1.01,
        iters: Optional[int] = 20,
        verbose: Optional[bool] = False,
    ) -> None:
        self.lp_norm = lp_norm
        self.beta = beta
        self.kappa = kappa
        self.iters = iters
        self.verbose = verbose

    def optimize(
        self, base: torch.Tensor, qtype: qtype, axis: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        scale, shift = super().optimize(base, qtype, axis)
        best_error = None
        beta = self.beta
        base_q = quantize_weight(base, qtype=qtype, axis=axis, scale=scale, shift=shift)
        for i in range(self.iters):
            error = base - base_q
            if best_error is None:
                best_error = float(torch.abs(base - base_q).mean())
                if self.verbose:
                    print(f"Start error: {best_error:.6f}")
            e = shrink_lp_op(error, beta, self.lp_norm)
            mean_axis = 0 if axis == -1 else -1
            hqq_shift = torch.mean(base_q._data * scale - (base - e), axis=mean_axis, keepdim=True)
            base_q = quantize_weight(base, qtype=qtype, axis=axis, scale=scale, shift=hqq_shift)
            mean_error = float(torch.abs(base - base_q).mean())
            if self.verbose:
                print(f"HQQ error at it #{i}: {mean_error:.6f}")
            if mean_error < best_error:
                best_error = mean_error
                shift = hqq_shift
                beta *= self.kappa
            else:
                break

        return scale, shift
