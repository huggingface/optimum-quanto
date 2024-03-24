from typing import Optional

import torch

from ..core import group
from .optimizer import Optimizer


__all__ = ["SymmetricOptimizer"]


class SymmetricOptimizer(Optimizer):

    def __call__(
        self, base: torch.Tensor, bits: int, axis: Optional[int] = None, group_size: Optional[int] = None
    ) -> torch.Tensor:
        if axis not in [None, 0, -1]:
            raise ValueError("axis parameter must be None, 0 (first axis) or -1 (last axis)")
        if group_size is not None:
            base = group(base, axis=axis, group_size=group_size)
        scale = self.optimize(base, bits, axis)
        assert scale.dtype == base.dtype
        return scale

    def optimize(self, base: torch.Tensor, bits: int, axis: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError
