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

from quanto import AWQBitsTensor, QBitsTensor, qint2, qint4, quantize_weight


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_qbitstensor_serialization(qtype, axis):
    qa = random_qweight((5, 5), qtype=qtype, axis=axis)
    b = io.BytesIO()
    torch.save(qa, b)
    b.seek(0)
    qa_reloaded = torch.load(b)
    assert isinstance(qa_reloaded, QBitsTensor)
    assert qa_reloaded.qtype == qa.qtype
    assert qa_reloaded.dtype == qa.dtype
    assert torch.equal(qa_reloaded._data, qa._data)
    assert torch.equal(qa_reloaded._scale, qa._scale)
    assert torch.equal(qa_reloaded._zeropoint, qa._zeropoint)


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [None, 16], ids=["channel-wise", "group-wise"])
def test_qbitstensor_requires_grad(qtype, axis, group_size, device):
    weight = random_tensor((32, 32), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = quantize_weight(weight, qtype=qtype, axis=axis, group_size=group_size)
    assert qweight.requires_grad is True


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [None, 16], ids=["channel-wise", "group-wise"])
def test_qbitstensor_backward(qtype, axis, group_size, device):
    weight = random_tensor((32, 32), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = quantize_weight(weight, qtype=qtype, axis=axis, group_size=group_size)
    gradient = torch.randn((32, 32)).to(device)
    # Backpropagate gradient to the inner float weights
    qweight.dequantize().backward(gradient)
    assert torch.equal(weight.grad, gradient)


@pytest.mark.parametrize("group_size", [None, 128], ids=["channel-wise", "group-wise"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_to_device(dtype, group_size, device):
    qa = random_qweight((256, 512), dtype=dtype, qtype=qint4, group_size=group_size, device="cpu")
    # Keep a copy of the dequantized Tensor as a reference
    dqa_cpu = qa.dequantize()
    # Move to the target device
    moved_qa = qa.to(device)
    assert isinstance(moved_qa, QBitsTensor)
    assert moved_qa.device.type == device.type
    assert moved_qa._data.device.type == device.type
    assert moved_qa._scale.device.type == device.type
    assert moved_qa._zeropoint.device.type == device.type
    if dtype == torch.float16 and device.type == "cuda" and group_size == 128:
        # For specific CUDA configurations, we use an optimized packing
        assert isinstance(moved_qa, AWQBitsTensor)
    # Verify the moved dequantized Tensor is identical
    moved_dqa = moved_qa.dequantize()
    assert torch.equal(moved_dqa.to("cpu"), dqa_cpu)


def test_detach():
    qa = random_qweight((32, 32), qtype=qint4)
    dqa = qa.detach()
    assert isinstance(dqa, QBitsTensor)
