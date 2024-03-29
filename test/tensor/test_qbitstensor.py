import io

import pytest
import torch
from helpers import random_qweight, random_tensor

from quanto import QBitsTensor, qint2, qint4, quantize_weight


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


def test_to_device(device):
    qa = random_qweight((32, 32), qtype=qint4)
    qa = qa.to(device)
    assert isinstance(qa, QBitsTensor)
    assert qa.device.type == device.type
    assert qa._data.device.type == device.type
    assert qa._scale.device.type == device.type
    assert qa._zeropoint.device.type == device.type


def test_detach():
    qa = random_qweight((32, 32), qtype=qint4)
    dqa = qa.detach()
    assert isinstance(dqa, QBitsTensor)
