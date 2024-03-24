from abc import ABC
from typing import Optional, Tuple, Union

import torch


__all__ = ["Optimizer"]


class Optimizer(ABC):

    def __call__(
        self, base: torch.Tensor, bits: int, axis: int, group_size: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError
