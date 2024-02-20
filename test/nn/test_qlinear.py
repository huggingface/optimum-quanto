import pytest
import torch
from helpers import assert_similar, random_qtensor

from quanto import Calibration, QBitsTensor, QTensor, qfloat8_e4m3fn, qfloat8_e5m2, qint4, qint8
from quanto.nn import QLinear


def _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, activations, dtype, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(dtype).to(device)
    qlinear = QLinear.from_module(linear, weights=weights, activations=activations)
    assert qlinear.qweight().qtype == weights
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=dtype).to(device)
    # Run an inference with Calibration to get the correct output dtype
    with torch.no_grad(), Calibration():
        qout = qlinear(qinputs)
    if activations is not None:
        assert isinstance(qout, QTensor)
        assert qout.qtype == activations
    # Align linear weights with quantized linear weights for comparison
    linear.weight = torch.nn.Parameter(qlinear.qweight().dequantize())
    out = linear(qinputs.dequantize())
    # We need to increase atol for float16 dtype
    dtype_atol = {torch.float32: 1e-4, torch.float16: 1e-3}[dtype]
    # We also need to increase atol for float8 qtypes
    atol = {None: dtype_atol, qint8: dtype_atol, qfloat8_e5m2: 5e-3, qfloat8_e4m3fn: 5e-3}[activations]
    assert_similar(out, qout, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
@pytest.mark.skip_device("cpu")
def test_quantize_linear_float16_activations_int8(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, qint8, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
def test_quantize_linear_float32_activations_int8(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, qint8, torch.float32, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn],
    ids=["a-qfloat8-e5m2", "a-qfloat8-e4m3"],
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
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn],
    ids=["a-qfloat8-e5m2", "a-qfloat8-e4m3"],
)
@pytest.mark.skip_device("mps")
def test_quantize_linear_float32_activations_float8(
    batch_size, tokens, embeddings, use_bias, weights, activations, device
):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, activations, torch.float32, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
@pytest.mark.skip_device("cpu")
def test_quantize_linear_float16_weight_only(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, None, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
def test_quantize_linear_float32_weight_only(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, None, torch.float32, device)


@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("activations", [None, qint8], ids=["a-float", "a-qint8"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
def test_qlinear_gradient(tokens, embeddings, activations, weights, device):
    batch_size = 10
    linear = torch.nn.Linear(embeddings, embeddings).to(device)
    qlinear = QLinear.from_module(linear, weights=weights, activations=activations)
    assert qlinear.weight.requires_grad is True
    assert qlinear.bias.requires_grad is True
    # Run an inference with identical inputs
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    qout = qlinear(qinputs)
    out = linear(qinputs.dequantize())
    # Outputs are not identical because of the quantization
    assert not torch.equal(qout, out)
    # Compute gradients and compare
    gradient = torch.randn(qout.size()).to(device)
    qout.backward(gradient)
    out.backward(gradient)
    # Gradients are identical because they depend only on the input
    assert torch.allclose(qlinear.weight.grad, linear.weight.grad)
    assert torch.allclose(qlinear.bias.grad, linear.bias.grad)


@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
def test_move_qlinear(use_bias, weights, device):
    linear = torch.nn.Linear(32, 32, bias=use_bias)
    qlinear = QLinear.from_module(linear, weights=weights)
    qlinear.freeze()
    qlinear.to(device)
    assert qlinear.weight._data.device.type == device.type
    assert qlinear.weight._scale.device.type == device.type
    if isinstance(qlinear.weight, QBitsTensor):
        assert qlinear.weight._zeropoint.device.type == device.type
