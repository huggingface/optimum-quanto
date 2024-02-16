import io

import pytest
import torch
from helpers import device_eq, q_assert_close, random_tensor

from quanto import QBitsTensor, qint2, qint4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("pack", [True, False], ids=["pack", "not-packed"])
def test_quantize_integer_tensor(dtype, qtype, device, pack):
    """This test verifies that an integer tensor in the correct range is preserved."""
    bits = qtype.bits
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    a = torch.tensor(range(qmin, qmax + 1), dtype=dtype).to(device)
    qa = QBitsTensor.quantize(a, qtype=qtype, pack=pack)

    assert qa._data.dtype == torch.uint8 if pack else torch.int8
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    assert torch.equal(a, qa.dequantize())


@pytest.mark.parametrize("input_shape", [(10,), (12,), (10, 10), (12, 10), (32, 32)])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("zp", [-1, 0, 1], ids=["neg", "centered", "pos"])
def test_quantize_per_tensor(input_shape, qtype, dtype, zp, device):
    a = random_tensor(input_shape, dtype=dtype).to(device) + zp
    qa = QBitsTensor.quantize(a, qtype=qtype)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    if input_shape[0] % (8 // qtype.bits) == 0:
        assert qa.packed
    q_assert_close(a, qa)


@pytest.mark.parametrize("axis", [0, 1, -1], ids=["first-axis", "second-axis", "last-axis"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("zp", [-1, 0, 1], ids=["neg", "centered", "pos"])
def test_quantize_per_axis(axis, qtype, dtype, zp, device):
    a = random_tensor((32, 32), dtype=dtype).to(device) + zp
    qa = QBitsTensor.quantize(a, qtype=qtype, axis=axis)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    q_assert_close(a, qa)


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["int2", "int4"])
@pytest.mark.parametrize("axis", [0, None, -1], ids=["first-axis", "per-tensor", "last-axis"])
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
