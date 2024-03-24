from typing import Optional, Tuple, Union

import torch

from .symmetric_optimizer import SymmetricOptimizer


__all__ = ["AbsmaxOptimizer"]


class AbsmaxOptimizer(SymmetricOptimizer):

    def optimize(
        self, base: torch.Tensor, bits: int, axis: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        base = torch.abs(base)
        if axis is None:
            rmax = torch.max(base)
        else:
            dim = list(range(1, base.ndim)) if (axis == 0) else list(range(0, base.ndim - 1))
            rmax = torch.amax(torch.abs(base), dim=dim, keepdim=True)
        qmax = 2 ** (bits - 1) - 1
        return rmax / qmax
