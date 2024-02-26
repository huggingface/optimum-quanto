import io

import pytest
import torch
from helpers import assert_similar, device_eq, random_tensor

from quanto import QBitsTensor, qint2, qint4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
def test_qbitstensor_quantize_integer_tensor(dtype, qtype, device):
    """This test verifies that an integer tensor in the correct range is preserved."""
    bits = qtype.bits
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    a = torch.tensor(range(qmin, qmax + 1), dtype=dtype).to(device)
    qa = QBitsTensor.quantize(a, qtype=qtype)

    assert qa._data.dtype == torch.uint8
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    assert torch.equal(a, qa.dequantize())


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("zp", [-1, 0, 1], ids=["neg", "centered", "pos"])
def test_qbitstensor_quantize_per_axis(axis, qtype, dtype, zp, device):
    a = random_tensor((32, 32), dtype=dtype).to(device) + zp
    qa = QBitsTensor.quantize(a, qtype=qtype, axis=axis)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    rtol = {qint4: 1e-2, qint2: 2e-1}[qtype]
    assert_similar(a, qa, rtol=rtol)


@pytest.mark.parametrize("input_shape", [(256, 256), (256, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [64, 128])
def test_qbitstensor_quantize_groupwise(input_shape, dtype, qtype, axis, group_size, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    qa = QBitsTensor.quantize(a, qtype=qtype, axis=axis, group_size=group_size)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert qa.shape == input_shape
    assert device_eq(qa.device, device)
    rtol = {qint4: 1e-2, qint2: 2e-1}[qtype]
    assert_similar(a, qa, rtol=rtol)


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_qbitstensor_serialization(qtype, axis):
    a = random_tensor((5, 5), dtype=torch.float32)
    qa = QBitsTensor.quantize(a, qtype=qtype, axis=axis)
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


def test_qbitstensor_requires_grad(device):
    weight = random_tensor((10,), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = QBitsTensor.quantize(weight)
    assert qweight.requires_grad is True


def test_qbitstensor_backward(device):
    weight = random_tensor((10,), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = QBitsTensor.quantize(weight)
    gradient = torch.randn((10,)).to(device)
    # Backpropagate gradient to the inner float weights
    qweight.dequantize().backward(gradient)
    assert torch.equal(weight.grad, gradient)
