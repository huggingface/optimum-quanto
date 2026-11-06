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

from ..grouped import group
from ..qtype import qtype
from .optimizer import Optimizer


__all__ = ["AffineOptimizer"]


class AffineOptimizer(Optimizer):
    def __call__(
        self,
        base: torch.Tensor,
        qtype: qtype,
        axis: int,
        group_size: Optional[int] = None,
        zeropoint: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base (`torch.Tensor`): the weight Tensor to quantize
            qtype (`quanto.qtype`): The target quantization type
            axis ('int`): The quantization axis (0 or -1)
            group_size (`Optional[int]`): The quantization group size
            zeropoint (`bool`): Allow an exact representation of zero. If True, the shifts are stored as
                integer instead of float, which results in a slightly smaller model, but might also reduce
                the model performance. When the optimized interval does not contain zero, the scale is
                increased so that zero is included in the representable integer interval. Defaults to False.
        Returns:
            A tuple of scale, shift Tensor.
        """
        if axis not in [0, -1]:
            raise ValueError("axis parameter must be 0 (first axis) or -1 (last axis)")
        if group_size is not None:
            base = group(base, axis, group_size)
        if axis is not None and base.shape[axis] == 1:
            axis = None
        scale, shift = self.optimize(base, qtype, axis)
        assert scale.dtype == base.dtype
        assert shift.dtype == base.dtype
        if zeropoint:
            # Include zero in one-sided optimized intervals before converting the shift.
            # Leave intervals that already contain zero unchanged.
            qmax = 2**qtype.bits - 1
            entirely_positive = shift < 0
            entirely_negative = shift > qmax * scale
            # Divide before subtracting to avoid overflowing the positive endpoint in float16.
            expanded_scale = torch.where(
                entirely_positive,
                scale - shift / qmax,
                torch.where(entirely_negative, shift / qmax, scale),
            )
            # Rounding a scale up can otherwise overflow the largest decoded value.
            overflow = (entirely_positive | entirely_negative) & torch.isinf(expanded_scale * qmax)
            expanded_scale = torch.where(
                overflow, torch.nextafter(expanded_scale, torch.zeros_like(expanded_scale)), expanded_scale
            )
            rounded_shift = torch.round(shift / scale)
            # Constant zero intervals have both scale and shift equal to zero.
            rounded_shift = torch.where(scale == 0, torch.zeros_like(rounded_shift), rounded_shift)
            shift = torch.where(
                entirely_positive,
                torch.zeros_like(shift),
                torch.where(entirely_negative, torch.full_like(shift, qmax), rounded_shift),
            )
            scale = expanded_scale
            shift = torch.clamp(shift, 0, qmax)
            shift = shift.to(torch.int8) if base.device.type == "xpu" else shift.to(torch.uint8)
        return scale, shift

    def optimize(self, base: torch.Tensor, qtype: qtype, axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
