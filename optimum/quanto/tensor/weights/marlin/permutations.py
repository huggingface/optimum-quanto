# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import List, Tuple

import torch

from ..reordering import reorder, reverse


__all__ = ["marlin_permute"]


# https://github.com/IST-DASLab/marlin/blob/2f6d7c10e124b3c5fa29ff8d77d568bd7af3274c/marlin/__init__.py#L40C1-L68C54
@functools.cache
def _get_perms() -> Tuple[List[int], List[int]]:
    perm = []
    for i in range(8):
        perm.extend([i + 8 * j for j in range(8)])
    perm_single = []
    for i in range(4):
        perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, perm_single


@functools.cache
def _get_inverted_perms() -> Tuple[List[int], List[int]]:
    perm, perm_single = _get_perms()
    return reverse(perm), reverse(perm_single)


def marlin_permute(t: torch.Tensor, reverse=False):
    perm, perm_single = _get_inverted_perms() if reverse else _get_perms()
    out_features = t.shape[1]
    if t.shape[0] == 1:
        reordered = reorder(t, perm_single)
    else:
        reordered = reorder(t, perm)
    return reordered.reshape((-1, out_features)).contiguous()
