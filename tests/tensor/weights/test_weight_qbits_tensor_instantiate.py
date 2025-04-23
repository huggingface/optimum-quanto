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

from optimum.quanto import qint2, qint4
from optimum.quanto.tensor.weights import WeightQBitsTensor


def random_data_scale_shift(input_shape, dtype, qtype, axis, group_size):
    out_features, in_features = input_shape
    n_groups = in_features * out_features // group_size
    data_shape = (n_groups, group_size) if axis == 0 else (group_size, n_groups)
    scale_shape = (n_groups, 1) if axis == 0 else (1, n_groups)
    min_value = -(2 ** (qtype.bits - 1))
    max_value = 2 ** (qtype.bits - 1) - 1
    data = torch.randint(max_value - min_value + 1, data_shape, dtype=torch.uint8)
    scale = torch.full(scale_shape, 1.0 / -min_value, dtype=dtype)
    shift = torch.ones(scale_shape, dtype=dtype)
    return data, scale, shift


@pytest.mark.parametrize("input_shape, group_size", [[(32, 32), 16], [(1024, 1024), 128]])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
def test_weight_qbits_tensor_instantiate(input_shape, dtype, qtype, axis, group_size, device):
    data, scale, shift = random_data_scale_shift(input_shape, dtype, qtype, axis, group_size)
    input_stride = torch.ones(input_shape).stride()
    qa = WeightQBitsTensor(qtype, axis, group_size, input_shape, input_stride, data, scale=scale, shift=shift).to(
        device
    )
    assert torch.max(torch.abs(qa.dequantize())) <= 1
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert qa.shape == input_shape


@pytest.mark.parametrize("input_shape, group_size", [[(32, 32), 16], [(1024, 1024), 128]])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
def test_weight_qbits_tensor_equal(input_shape, dtype, qtype, axis, group_size, device):
    data, scale, shift = random_data_scale_shift(input_shape, dtype, qtype, axis, group_size)
    qa = WeightQBitsTensor(qtype, axis, group_size, data.size(), data.stride(), data, scale=scale, shift=shift).to(
        device
    )
    qb = WeightQBitsTensor(
        qtype, axis, group_size, data.size(), data.stride(), data.clone(), scale=scale.clone(), shift=shift.clone()
    ).to(device)
    assert qa.equal(qb)
