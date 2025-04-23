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

import io

import pytest
import torch
from helpers import random_qweight, random_tensor

from optimum.quanto import MaxOptimizer, WeightQBitsTensor, qint2, qint4, quantize_weight


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_weight_qbits_tensor_serialization(qtype, axis):
    qa = random_qweight((5, 5), qtype=qtype, axis=axis)
    b = io.BytesIO()
    torch.save(qa, b)
    b.seek(0)
    qa_reloaded = torch.load(b, weights_only=False)
    assert isinstance(qa_reloaded, WeightQBitsTensor)
    assert qa_reloaded.qtype == qa.qtype
    assert qa_reloaded.dtype == qa.dtype
    assert torch.equal(qa_reloaded._data, qa._data)
    assert torch.equal(qa_reloaded._scale, qa._scale)
    assert torch.equal(qa_reloaded._shift, qa._shift)


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [None, 16], ids=["channel-wise", "group-wise"])
def test_weight_qbits_tensor_requires_grad(qtype, axis, group_size, device):
    weight = random_tensor((32, 32), dtype=torch.float32).to(device)
    weight.requires_grad = True
    scale, shift = MaxOptimizer()(weight, qtype=qtype, axis=axis, group_size=group_size)
    qweight = quantize_weight(weight, qtype=qtype, axis=axis, scale=scale, shift=shift, group_size=group_size)
    assert qweight.requires_grad is True


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [None, 16], ids=["channel-wise", "group-wise"])
def test_weight_qbits_tensor_backward(qtype, axis, group_size, device):
    weight = random_tensor((32, 32), dtype=torch.float32).to(device)
    weight.requires_grad = True
    scale, shift = MaxOptimizer()(weight, qtype=qtype, axis=axis, group_size=group_size)
    qweight = quantize_weight(weight, qtype=qtype, axis=axis, scale=scale, shift=shift, group_size=group_size)
    gradient = torch.randn((32, 32)).to(device)
    # Backpropagate gradient to the inner float weights
    qweight.dequantize().backward(gradient)
    assert torch.equal(weight.grad, gradient)
