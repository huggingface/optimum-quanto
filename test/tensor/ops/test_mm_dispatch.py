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
from helpers import assert_similar, random_qactivation, random_qweight

from optimum.quanto import qint8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("in_features", [5, 16, 24])
@pytest.mark.parametrize("hidden", [5, 16, 24])
@pytest.mark.parametrize("out_features", [5, 16, 24])
def test_qactivation_qweight_matmul(dtype, in_features, hidden, out_features, device):
    qa = random_qactivation((in_features, hidden), qint8, dtype=dtype).to(device)
    qb = random_qweight((hidden, out_features), qint8, dtype=dtype, axis=-1).to(device)
    qmatmul = torch.matmul(qa, qb)
    # The outputs should be almost identical if we use the dequantized inputs
    matmul = torch.matmul(qa.dequantize(), qb.dequantize())
    assert_similar(matmul, qmatmul)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("a_shape, b_shape", [[(16, 32), (32, 24)], [(5, 10), (10, 6)]])
def test_qactivation_qactivation_bmm(dtype, batch_size, a_shape, b_shape, device):
    qa = random_qactivation((batch_size,) + a_shape, qint8, dtype=dtype).to(device)
    qb = random_qactivation((batch_size,) + b_shape, qint8, dtype=dtype).to(device)
    qbmm = torch.bmm(qa, qb)
    # The outputs should be almost identical if we use the dequantized inputs
    bmm = torch.bmm(qa.dequantize(), qb.dequantize())
    assert_similar(bmm, qbmm)
