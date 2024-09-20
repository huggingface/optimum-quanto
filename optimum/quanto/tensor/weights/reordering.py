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

from typing import List, Union

import torch


__all__ = ["reorder", "reverse"]


def reorder(t: torch.Tensor, permutation: Union[torch.Tensor, List[int]]):
    """Reorder a Tensor using a permutation

    Args:
        t (`torch.Tensor`): the Tensor to reorder
        permutation (`Union[torch.Tensor, List[int]]`): the permutation to apply

    Returns:
        The reordered torch.Tensor
    """
    block_size = permutation.numel() if isinstance(permutation, torch.Tensor) else len(permutation)
    reordered = t.reshape((-1, block_size))[:, permutation].reshape(t.shape)
    return reordered.contiguous()


def reverse(permutation: Union[torch.Tensor, List[int]]):
    """Reverse a permutation

    The reversed permutation can be used to revert a reordered Tensor to its original
    ordering.

    Args:
        permutation (`Union[torch.Tensor, List[int]]`): the permutation to reverse

    Returns:
        The reversed permutation
    """
    block_size = permutation.numel() if isinstance(permutation, torch.Tensor) else len(permutation)
    reversed = torch.empty((block_size,), dtype=torch.int64)
    reversed[permutation] = torch.arange(block_size)
    return reversed
