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

import pytest
import torch
from helpers import random_qweight

from quanto import qint8


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize(
    "group_size",
    [None, 2],
    ids=["channel-wise", "group-wise"],
)
def test_qweight_transpose_2d(axis, group_size, device):
    input_shape = (4, 6)
    qinputs = random_qweight(input_shape, qint8, axis=axis, group_size=group_size).to(device)
    qtransposed = qinputs.t()
    assert qtransposed.qtype == qinputs.qtype
    if axis == -1:
        assert qtransposed.axis == 0
    elif axis == 0:
        assert qtransposed.axis == -1
    assert qtransposed.shape == input_shape[::-1]
    assert torch.equal(qtransposed.dequantize(), qinputs.dequantize().t())
