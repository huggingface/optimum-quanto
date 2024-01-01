import io

import pytest
import torch
from helpers import device_eq, q_assert_close, random_tensor

from quanto import QBitsTensor


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
def test_quantize_integer_tensor(bits, dtype, device):
    """This test verifies that an integer tensor in the correct range is preserved."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    a = torch.tensor(range(qmin, qmax + 1), dtype=dtype).to(device)
    qa = QBitsTensor.quantize(a, bits=bits)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert device_eq(qa.device, device)
    assert torch.equal(a, qa.dequantize())


@pytest.mark.parametrize("input_shape", [(10,), (10, 10), (32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
@pytest.mark.parametrize("zp", [-1, 0, 1], ids=["neg", "centered", "pos"])
def test_quantize_per_tensor(input_shape, bits, dtype, zp, device):
    a = random_tensor(input_shape, dtype=dtype).to(device) + zp
    qa = QBitsTensor.quantize(a, bits=bits)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert device_eq(qa.device, device)
    q_assert_close(a, qa)


@pytest.mark.parametrize("axis", [0, 1, -1], ids=["first-axis", "second-axis", "last-axis"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
@pytest.mark.parametrize("zp", [-1, 0, 1], ids=["neg", "centered", "pos"])
def test_quantize_per_axis(axis, bits, dtype, zp, device):
    a = random_tensor((32, 32), dtype=dtype).to(device) + zp
    qa = QBitsTensor.quantize(a, bits=bits, axis=axis)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert device_eq(qa.device, device)
    q_assert_close(a, qa)


@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, None, -1], ids=["first-axis", "per-tensor", "last-axis"])
def test_qbitstensor_serialization(bits, axis):
    a = random_tensor((5, 5), dtype=torch.float32)
    qa = QBitsTensor.quantize(a, bits=bits, axis=axis)
    b = io.BytesIO()
    torch.save(qa, b)
    b.seek(0)
    qa_reloaded = torch.load(b)
    assert isinstance(qa_reloaded, QBitsTensor)
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
