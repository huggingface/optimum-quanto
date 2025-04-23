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
from helpers import random_tensor

from optimum.quanto import (
    HqqOptimizer,
    MaxOptimizer,
    WeightQBitsTensor,
    qint2,
    qint4,
)


def compare_quantized_tensor(a, qtype, axis, group_size, scale, shift):
    qa = WeightQBitsTensor.quantize(a, qtype, axis, group_size, scale, shift)
    # Evaluate mean absolute error
    mean_error = torch.mean(torch.abs(a - qa))
    # Also evaluate cosine similarity
    sim = torch.nn.functional.cosine_similarity(a.flatten(), qa.flatten(), dim=0)
    return mean_error, sim


@pytest.mark.parametrize("input_shape", [(1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_hqq_optimizer(input_shape, dtype, qtype, axis, group_size, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    max_scale, max_shift = MaxOptimizer()(a, qtype=qtype, axis=axis, group_size=group_size)
    max_mean_error, max_sim = compare_quantized_tensor(a, qtype, axis, group_size, max_scale, max_shift)
    hqq_scale, hqq_shift = HqqOptimizer()(a, qtype=qtype, axis=axis, group_size=group_size)
    hqq_mean_error, hqq_sim = compare_quantized_tensor(a, qtype, axis, group_size, hqq_scale, hqq_shift)
    # HQQ optimizes the mean error, so it should be lower
    assert hqq_mean_error <= max_mean_error
    # FIXME: HQQ cosine similarity should be also closer to 1
