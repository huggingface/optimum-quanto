import io
from math import prod

import pytest
import torch
from helpers import assert_similar, device_eq, random_qtensor, random_tensor

from quanto import QTensor, absmax_scale, qfloat8, qfloat8_e4m3fn, qfloat8_e5m2, qint8


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_qtensor_quantize_int8(input_shape, dtype, device):
    qtype = qint8
    a = random_tensor(input_shape, dtype=dtype).to(device)
    qa = QTensor.quantize(a, qtype=qtype)
    assert isinstance(qa, QTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    assert_similar(a, qa)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qfloat8_e5m2, qfloat8_e4m3fn], ids=["qfloat8_e5m2", "qfloat8_e4m3"])
@pytest.mark.skip_device("mps")
def test_qtensor_quantize_float8(input_shape, dtype, qtype, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    qa = QTensor.quantize(a, qtype=qtype)
    assert isinstance(qa, QTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    assert_similar(a, qa, atol=5e-3)


@pytest.mark.parametrize("input_shape", [(32, 64), (10, 32, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_qtensor_quantize_per_axis(input_shape, dtype, qtype, axis, device):
    if device.type == "mps" and qtype.is_floating_point:
        pytest.skip("Float8 is not supported on MPS device")
    a = random_tensor(input_shape, dtype=dtype).to(device)
    qa = QTensor.quantize(a, qtype=qtype, axis=axis)
    assert isinstance(qa, QTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    assert_similar(a, qa, atol=5e-3)


@pytest.mark.parametrize("input_shape", [(256, 256), (256, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [64, 128])
def test_qtensor_quantize_groupwise(input_shape, dtype, qtype, axis, group_size, device):
    if device.type == "mps" and qtype.is_floating_point:
        pytest.skip("Float8 is not supported on MPS device")
    a = random_tensor(input_shape, dtype=dtype).to(device)
    qa = QTensor.quantize(a, qtype=qtype, axis=axis, group_size=group_size)
    assert isinstance(qa, QTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert qa.shape == input_shape
    assert device_eq(qa.device, device)
    assert_similar(a, qa, atol=5e-3)


@pytest.mark.parametrize("input_shape", [(10, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8], ids=["qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_qtensor_quantize_scale(input_shape, axis, dtype, qtype, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, qtype, axis)
    qa = QTensor.quantize(a, qtype=qtype, axis=axis, group_size=None, scale=scale)
    if axis is not None:
        if a.shape[axis] == 1:
            # Quantization is actually per-tensor as the axis dim is 1
            assert qa.axis is None
        else:
            assert qa.axis == axis
    assert isinstance(qa, QTensor)
    assert qa.qtype == qtype
    assert qa._scale.dtype == dtype
    assert device_eq(qa.device, device)
    assert_similar(a, qa)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
def test_qtensor_instantiate(input_shape, dtype, qtype, device):
    if qtype.is_floating_point:
        if device.type == "mps":
            pytest.skip("float8 types are not supported on MPS device")
        min_value = torch.finfo(qtype.dtype).min
        max_value = torch.finfo(qtype.dtype).max
        data = (torch.rand(input_shape) * max_value + min_value).to(qtype.dtype)
    else:
        max_value = torch.iinfo(qtype.dtype).max
        data = torch.randint(-max_value, max_value, input_shape, dtype=qtype.dtype)
    qa = QTensor(qtype, None, data.size(), data.stride(), data, scale=torch.tensor(1.0 / max_value, dtype=dtype)).to(
        device
    )
    assert torch.max(torch.abs(qa.dequantize())) <= 1
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert qa.shape == input_shape


@pytest.mark.parametrize("input_shape", [(10, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_qtensor_serialization(input_shape, qtype, dtype, axis):
    inputs = random_tensor(input_shape, dtype=dtype)
    scale = absmax_scale(inputs, qtype, axis)
    qinputs = QTensor.quantize(inputs, qtype=qtype, axis=axis, group_size=None, scale=scale)
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


def test_qtensor_requires_grad(device):
    weight = random_tensor((10,), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = QTensor.quantize(weight)
    assert qweight.requires_grad is True


def test_qtensor_backward(device):
    weight = random_tensor((10,), dtype=torch.float32).to(device)
    weight.requires_grad = True
    qweight = QTensor.quantize(weight)
    gradient = torch.randn((10,)).to(device)
    # Backpropagate gradient to the inner float weights
    qweight.dequantize().backward(gradient)
    assert torch.equal(weight.grad, gradient)


def test_qtensor_chained_backward(device):
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


@pytest.mark.parametrize("input_shape, group_size", [[(4, 6), 2], [(32, 64), 4]], ids=["small", "bigger"])
def test_qtensor_quantize_transposed_groupwise(input_shape, group_size, device):
    x = torch.tensor(range(prod(input_shape))).reshape(input_shape).to(device)
    xt = x.t()
    qx = QTensor.quantize(x, axis=0, group_size=group_size)
    qxt = QTensor.quantize(xt, axis=-1, group_size=group_size)
    dqx = qx.dequantize()
    dqxt = qxt.dequantize()
    assert torch.equal(dqx.t(), dqxt)
