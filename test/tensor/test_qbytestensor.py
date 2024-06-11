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

from optimum.quanto import QBytesTensor, qfloat8, qint8, quantize_weight


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
def test_qbytestensor_instantiate(input_shape, dtype, qtype, device):
    if qtype.is_floating_point:
        if device.type == "mps":
            pytest.skip("float8 types are not supported on MPS device")
        min_value = torch.finfo(qtype.dtype).min
        max_value = torch.finfo(qtype.dtype).max
        data = (torch.rand(input_shape) * max_value + min_value).to(qtype.dtype)
    else:
        max_value = torch.iinfo(qtype.dtype).max
        data = torch.randint(-max_value, max_value, input_shape, dtype=qtype.dtype)
    qa = QBytesTensor(
        qtype, None, data.size(), data.stride(), data, scale=torch.tensor(1.0 / max_value, dtype=dtype)
    ).to(device)
    assert torch.max(torch.abs(qa.dequantize())) <= 1
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert qa.shape == input_shape


@pytest.mark.parametrize("input_shape", [(10, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_qbytestensor_serialization(input_shape, qtype, dtype, axis):
    qinputs = random_qweight(input_shape, dtype=dtype, qtype=qtype, axis=axis)
    b = io.BytesIO()
    torch.save(qinputs, b)
    b.seek(0)
    qinputs_reloaded = torch.load(b)
    assert qinputs_reloaded.qtype == qtype
    assert torch.equal(qinputs_reloaded._scale, qinputs._scale)
    if qtype.is_floating_point:
        # Equality is not supported for float8
        assert torch.equal(qinputs_reloaded._data.to(torch.float32), qinputs._data.to(torch.float32))
    else:
        assert torch.equal(qinputs_reloaded._data, qinputs._data)
    # We cannot test dtype directly as it is not correctly set by torch.load
    assert qinputs_reloaded._scale.dtype == dtype
    assert qinputs_reloaded.axis == qinputs.axis


def test_qbytestensor_requires_grad(device):
    w = random_tensor((10, 10), dtype=torch.float32).to(device)
    w.requires_grad = True
    qw = quantize_weight(w, qtype=qint8, axis=0)
    assert qw.requires_grad is True


def test_qbytestensor_backward(device):
    w = random_tensor((10, 10), dtype=torch.float32).to(device)
    w.requires_grad = True
    qw = quantize_weight(w, qtype=qint8, axis=0)
    gradient = torch.randn((10, 10)).to(device)
    # Backpropagate gradient to the inner float weights
    qw.dequantize().backward(gradient)
    assert torch.equal(w.grad, gradient)


def test_qbytestensor_chained_backward(device):
    a = random_tensor((10, 10), dtype=torch.float32).to(device)
    a.requires_grad = True
    qa = quantize_weight(a, qtype=qint8, axis=0)
    b = random_tensor((10, 10), dtype=torch.float32).to(device)
    b.requires_grad = True
    qb = quantize_weight(b, qtype=qint8, axis=0)
    # Evaluate the product
    prod = qa * qb
    # Backpropagate
    gradient = torch.randn((10, 10)).to(device)
    prod.backward(gradient)
    assert torch.allclose(a.grad, qb.dequantize() * gradient)
    assert torch.allclose(b.grad, qa.dequantize() * gradient)


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("qtype", [qint8])
def test_qbytestensor_stride(axis, qtype, device):
    input_shape = (2, 4, 8)
    a = random_tensor(input_shape, dtype=torch.float32).to(device)
    qa = quantize_weight(a, qtype=qtype, axis=axis)
    assert qa.stride() == a.stride()
    ta = a.transpose(2, 1)
    tqa = qa.transpose(2, 1)
    assert tqa.stride() == ta.stride()


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("qtype", [qint8])
def test_qbytestensor_contiguous(axis, qtype, device):
    input_shape = (2, 4, 8)
    qa = random_qweight(input_shape, axis=axis, qtype=qtype, dtype=torch.float32).to(device)
    assert qa.is_contiguous()
    tqa = qa.transpose(2, 1)
    assert not tqa.is_contiguous()
    tqa = tqa.contiguous()
    assert tqa.is_contiguous()


def test_to_device(device):
    qa = random_qweight((32, 32), qtype=qint8, dtype=torch.float)
    qa = qa.to(device)
    assert isinstance(qa, QBytesTensor)
    assert qa.device.type == device.type
    assert qa._data.device.type == device.type
    assert qa._scale.device.type == device.type


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_quantize_weight_axis_dim_1(axis, device):
    input_shape = (1, 32) if axis == 0 else (32, 1)
    a = random_tensor(input_shape, dtype=torch.float32).to(device)
    qa = quantize_weight(a, qtype=qint8, axis=axis)
    # Quantizing along an axis of dimension 1 actually means per-tensor
    assert qa.axis is None
