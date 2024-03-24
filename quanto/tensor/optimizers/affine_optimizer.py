from typing import Optional, Tuple

import torch

from ..core import group
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
