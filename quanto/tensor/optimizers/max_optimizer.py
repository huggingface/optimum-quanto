from typing import Tuple, Union

import torch

from .affine_optimizer import AffineOptimizer


__all__ = ["MaxOptimizer"]


class MaxOptimizer(AffineOptimizer):

    def optimize(
        self, base: torch.Tensor, bits: int, axis: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dim = list(range(1, base.ndim)) if (axis == 0) else list(range(0, base.ndim - 1))
        rmin = torch.amin(base, dim=dim, keepdim=True)
        rmax = torch.amax(base, dim=dim, keepdim=True)
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = (rmax - rmin) / (qmax - qmin)
        zeropoint = torch.round(-rmin / scale).to(torch.int8)
        return scale, zeropoint
