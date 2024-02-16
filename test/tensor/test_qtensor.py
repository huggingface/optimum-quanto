import os
from tempfile import TemporaryDirectory

import pytest
import torch
from helpers import assert_similar, device_eq, q_assert_close, random_qtensor, random_tensor

from quanto import QTensor, absmax_scale, qfloat8_e4m3fn, qfloat8_e5m2, qint8, qint16, qint32


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8], ids=["qint8"])
def test_quantize_integer(input_shape, dtype, qtype, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    qa = QTensor.quantize(a, qtype)
    assert isinstance(qa, QTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    q_assert_close(a, qa)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qfloat8_e5m2, qfloat8_e4m3fn], ids=["qfloat8_e5m2", "qfloat8_e4m3"])
@pytest.mark.skip_device("mps")
def test_quantize_float8(input_shape, dtype, qtype, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    qa = QTensor.quantize(a, qtype)
    assert isinstance(qa, QTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    assert_similar(a, qa, atol=5e-3)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (2, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8], ids=["qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_quantize_scale(input_shape, axis, dtype, qtype, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, qtype, axis)
    qa = QTensor.quantize(a, qtype, scale)
    if axis is not None:
        if a.ndim == 1:
            # Quantization is actually per-tensor since the input tensor is a vector
            assert qa.axis is None
        elif a.shape[axis] == 1:
            # Quantization is actually per-tensor as the axis dim is 1
            assert qa.axis is None
        else:
            assert qa.axis == axis
    assert isinstance(qa, QTensor)
    assert qa.qtype == qtype
    assert qa._scale.dtype == dtype
    assert device_eq(qa.device, device)
    q_assert_close(a, qa)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8, qint16, qint32], ids=["qint8", "qint16", "qint32"])
def test_instantiate(input_shape, dtype, qtype, device):
    max_value = min(1024, torch.iinfo(qtype.dtype).max)
    data = torch.randint(-max_value, max_value, input_shape, dtype=qtype.dtype)
    qa = QTensor(qtype, data, scale=torch.tensor(1.0 / max_value, dtype=dtype)).to(device)
    assert torch.max(torch.abs(qa.dequantize())) <= 1
    assert qa.dtype == dtype
    assert qa.qtype == qtype


def test_quantized_tensor_serialization():
    qinputs = random_qtensor((1, 10, 32), dtype=torch.float32)
    with TemporaryDirectory() as tmpdir:
        qinputs_file = os.path.join(tmpdir, "qinputs.pt")
        torch.save(qinputs, qinputs_file)
        qinputs_reloaded = torch.load(qinputs_file)
    assert torch.equal(qinputs._data, qinputs_reloaded._data)
    assert torch.equal(qinputs._scale, qinputs_reloaded._scale)


def test_quantized_tensor_requires_grad(device):
    weight = random_tensor((10,), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = QTensor.quantize(weight)
    assert qweight.requires_grad is True


def test_quantized_tensor_backward(device):
    weight = random_tensor((10,), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = QTensor.quantize(weight)
    gradient = torch.randn((10,)).to(device)
    # Backpropagate gradient to the inner float weights
    qweight.dequantize().backward(gradient)
    assert torch.equal(weight.grad, gradient)


def test_quantized_tensor_chained_backward(device):
    a = random_tensor((10,), dtype=torch.float32).to(device)
    a.requires_grad = True
    qa = QTensor.quantize(a)
    b = random_tensor((10,), dtype=torch.float32).to(device)
    b.requires_grad = True
    qb = QTensor.quantize(b)
    # Evaluate the product
    prod = qa * qb
    # Backpropagate
    gradient = torch.randn((10,)).to(device)
    prod.backward(gradient)
    assert torch.allclose(a.grad, qb.dequantize() * gradient)
    assert torch.allclose(b.grad, qa.dequantize() * gradient)


def test_qtensor_stride(device):
    input_shape = (2, 4, 8)
    a = random_tensor(input_shape, dtype=torch.float32).to(device)
    qa = QTensor.quantize(a)
    assert qa.stride() == a.stride()
    ta = a.transpose(2, 1)
    tqa = qa.transpose(2, 1)
    assert tqa.stride() == ta.stride()


def test_qtensor_contiguous(device):
    input_shape = (2, 4, 8)
    qa = random_qtensor(input_shape, dtype=torch.float32).to(device)
    assert qa.is_contiguous()
    tqa = qa.transpose(2, 1)
    assert not tqa.is_contiguous()
    tqa = tqa.contiguous()
    assert tqa.is_contiguous()
