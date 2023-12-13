import pytest
import torch
from helpers import assert_similar, random_qtensor

from quanto.quantization import QTensor, calibration
from quanto.quantization.nn import QLinear


def _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, activations, dtype, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(dtype).to(device)
    qlinear = QLinear.from_module(linear, weights=weights, activations=activations)
    assert qlinear.qweight().itype == weights
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=dtype).to(device)
    # Run an inference with calibration to get the correct output dtype
    with torch.no_grad(), calibration():
        qout = qlinear(qinputs)
    if activations is not None:
        assert isinstance(qout, QTensor)
        assert qout.itype == activations
    # Align linear weights with quantized linear weights for comparison
    linear.weight = torch.nn.Parameter(qlinear.qweight().dequantize())
    out = linear(qinputs.dequantize())
    # We need to increase atol for float16 dtype
    dtype_atol = {torch.float32: 1e-4, torch.float16: 1e-3}[dtype]
    # We also need to increase atol for float8 itypes
    atol = {None: dtype_atol, torch.int8: dtype_atol, torch.float8_e5m2: 5e-3, torch.float8_e4m3fn: 5e-3}[activations]
    assert_similar(out, qout, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
@pytest.mark.skip_device("cpu")
def test_quantize_linear_float16_activations_int8(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, torch.int8, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
def test_quantize_linear_float32_activations_int8(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, torch.int8, torch.float32, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
@pytest.mark.parametrize(
    "activations",
    [torch.float8_e5m2, torch.float8_e4m3fn],
    ids=["a-float8-e5m2", "a-float8-e4m3"],
)
@pytest.mark.skip_device("cpu")
@pytest.mark.skip_device("mps")
def test_quantize_linear_float16_activations_float8(
    batch_size, tokens, embeddings, use_bias, weights, activations, device
):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, activations, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
@pytest.mark.parametrize(
    "activations",
    [torch.float8_e5m2, torch.float8_e4m3fn],
    ids=["a-float8-e5m2", "a-float8-e4m3"],
)
@pytest.mark.skip_device("mps")
def test_quantize_linear_float32_activations_float8(
    batch_size, tokens, embeddings, use_bias, weights, activations, device
):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, activations, torch.float32, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
@pytest.mark.skip_device("cpu")
def test_quantize_linear_float16_weight_only(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, None, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [torch.int8], ids=["w-int8"])
def test_quantize_linear_float32_weight_only(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, None, torch.float32, device)


@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("activations", [None, torch.int8], ids=["a-float", "a-int8"])
def test_qlinear_gradient(tokens, embeddings, activations, device):
    # We use a batch size of 1 to simplify gradient manual calculations
    batch_size = 1
    linear = torch.nn.Linear(embeddings, embeddings).to(device)
    qlinear = QLinear.from_module(linear, activations=activations)
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
