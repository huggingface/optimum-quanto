from typing import Tuple

import torch
from torch import Tensor

from quanto.tensor.core import axis_to_dim

from .affine_optimizer import AffineOptimizer
from .symmetric_optimizer import SymmetricOptimizer
from typing import Union, Optional


def _fake_quantize(base: Tensor, scale: Tensor, zeropoint: Union[int, Tensor], qmin: int, qmax: int) -> Tensor:
    data = base / scale + zeropoint
    data = torch.round(data)
    data = torch.clamp(data, qmin, qmax)
    data = (data - zeropoint) * scale
    return data


class MseSymmetricOptimizer(SymmetricOptimizer):
    def __init__(self, p: float = 2.0, iters: int = 80) -> None:
        self.iters = iters
        self.p = p
        assert p >= 1.0
        assert 0 < iters < 100

    def optimize(self, base: Tensor, bits: int, axis: int) -> Tensor:
        dim = None if axis is None else axis_to_dim(base, axis)
        rmax = torch.amax(torch.abs(base), dim=dim, keepdim=True)
        qmax = 2**bits - 1
        best_score = float("inf")
        best_rmax = rmax
        for i in range(self.iters):
            new_rmax = best_rmax * (1 - (i * 0.001))
            scale = new_rmax / qmax
            fq_base = _fake_quantize(base, scale, 0, -qmax, qmax)
            score = (fq_base - base).abs().pow(self.p).mean(dim) if axis else (fq_base - base).abs().pow(self.p).mean()
            if score.item() < best_score:
                best_score = score.item()
                best_rmax = new_rmax
        scale = best_rmax / qmax
        return scale


# QModuleMixin didn't support optimizer for activation yet, we make this internal temporarily
class _MseAffineOptimizer(AffineOptimizer):
    def __init__(self, p: float = 2.0, iters: int = 80) -> None:
        self.iters = iters
        self.p = p
        assert p >= 1.0
        assert 0 < iters < 100

    def optimize(self, base: Tensor, bits: int, axis: int) -> Tuple[Tensor, Tensor]:
        best_score = float("inf")
        dim = None if axis is None else axis_to_dim(base, axis)
        rmin, rmax = torch.aminmax(base, dim=dim, keepdim=True)
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        best_rmin, best_rmax = rmin, rmax
        for i in range(self.iters):
            new_rmin = best_rmin * (1 - (i * 0.001))
            new_rmax = best_rmax * (1 - (i * 0.001))
            scale = (new_rmax - new_rmin) / (qmax - qmin)
            zeropoint = torch.round(-rmin / scale).to(torch.int8)
            fq_base = _fake_quantize(base, scale, zeropoint, qmin, qmax)
            score = (fq_base - base).abs().pow(self.p).mean(dim) if axis else (fq_base - base).abs().pow(self.p).mean()
            if score.item() < best_score:
                best_score = score.item()
                best_rmin, best_rmax = new_rmin, new_rmax
        scale = (best_rmax - best_rmin) / (qmax - qmin)
        zeropoint = torch.round(-best_rmin / scale).to(torch.uint8)
        return scale, zeropoint
