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

from optimum.quanto import WeightQBytesTensor, qfloat8, qint8


def random_data_scale(input_shape, dtype, qtype):
    if qtype.is_floating_point:
        min_value = torch.finfo(qtype.dtype).min
        max_value = torch.finfo(qtype.dtype).max
        data = (torch.rand(input_shape) * max_value + min_value).to(qtype.dtype)
    else:
        max_value = torch.iinfo(qtype.dtype).max
        data = torch.randint(-max_value, max_value, input_shape, dtype=qtype.dtype)
    scale = torch.tensor(1.0 / max_value, dtype=dtype)
    return data, scale


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
def test_qbytestensor_instantiate(input_shape, dtype, qtype, device):
    if qtype.is_floating_point and device.type == "mps":
        pytest.skip("float8 types are not supported on MPS device")
    data, scale = random_data_scale(input_shape, dtype, qtype)
    qa = WeightQBytesTensor(qtype, None, data.size(), data.stride(), data, scale=scale, activation_qtype=None).to(
        device
    )
    assert torch.max(torch.abs(qa.dequantize())) <= 1
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert qa.shape == input_shape


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8], ids=["qint8"])
def test_qbytestensor_equal(input_shape, dtype, qtype, device):
    data, scale = random_data_scale(input_shape, dtype, qtype)
    qa = WeightQBytesTensor(qtype, None, data.size(), data.stride(), data, scale=scale, activation_qtype=None).to(
        device
    )
    qb = WeightQBytesTensor(
        qtype, None, data.size(), data.stride(), data.clone(), scale=scale, activation_qtype=None
    ).to(device)
    assert qa.equal(qb)
