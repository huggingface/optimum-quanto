import pytest
import torch
from helpers import q_assert_close, random_qtensor, random_tensor

from quanto.quantization import QTensor, calibration, freeze
from quanto.quantization.nn import QLinear


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("per_axis", [True, False], ids=["per-axis", "per-tensor"])
def test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, dtype, per_axis, device):
    if dtype == torch.float16 and device == torch.device("cpu"):
        pytest.skip("torch.ops.aten.addmm is not supported for float16 on CPU.")
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(dtype).to(device)
    qlinear = QLinear.from_module(linear, weights=weights)
    assert qlinear.qweight().itype == weights
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=dtype).to(device)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), calibration(per_axis=per_axis):
        qout = qlinear(qinputs)
    assert qout.itype == torch.int8
    # Freeze to set quantized weights
    freeze(qlinear)
    # Align linear weights with quantized linear weights for comparison
    linear.weight = torch.nn.Parameter(qlinear.weight.dequantize())
    out = linear(qinputs.dequantize())
    q_assert_close(out, qout)
    # Now run an inference with frozen model
    with torch.no_grad():
        int_qout = qlinear(qinputs)
    assert torch.equal(qout._scale, int_qout._scale)
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_quantize_linear_weight_only(batch_size, tokens, embeddings, use_bias, weights, dtype, device):
    if dtype == torch.float16 and device == torch.device("cpu"):
        pytest.skip("torch.ops.aten.addmm is not supported for float16 on CPU.")
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(dtype).to(device)
    qlinear = QLinear.from_module(linear, weights=weights)
    assert qlinear.qweight().itype == weights
    freeze(qlinear)
    inputs = random_tensor((batch_size,) + (tokens, embeddings), dtype=dtype).to(device)
    qout = qlinear(inputs)
    assert qout.dtype == dtype
    assert not isinstance(qout, QTensor)
    # Align linear weights with quantized linear weights for comparison
    linear.weight = torch.nn.Parameter(qlinear.weight.dequantize())
    out = linear(inputs)
    # We need to use higher values than the default that correspond to float64
    atol = {torch.float32: 1e-6, torch.float16: 1e-3}[dtype]
    rtol = {torch.float32: 1e-5, torch.float16: 1e-2}[dtype]
    assert torch.allclose(out, qout, rtol=rtol, atol=atol)


@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
def test_qlinear_gradient(tokens, embeddings, device):
    # We use a batch size of 1 to simplify gradient manual calculations
    batch_size = 1
    linear = torch.nn.Linear(embeddings, embeddings).to(device)
    qlinear = QLinear.from_module(linear)
    assert qlinear.weight.requires_grad is True
    assert qlinear.bias.requires_grad is True
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    qout = qlinear(qinputs)
    gradient = torch.randn(qout.size()).to(device)
    qout.backward(gradient)
    # Compute gradients manually and compare
    bias_gradient = torch.sum(gradient, axis=[0, 1])
    assert torch.allclose(qlinear.bias.grad, bias_gradient)
    weight_gradient = torch.matmul(gradient.squeeze().t(), qinputs.dequantize().squeeze())
    assert torch.allclose(qlinear.weight.grad, weight_gradient)
