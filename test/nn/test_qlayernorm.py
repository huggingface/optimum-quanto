import pytest
import torch
from helpers import assert_similar, random_qtensor

from quanto import Calibration, QTensor, qfloat8_e4m3fn, qfloat8_e5m2, qint8
from quanto.nn import QLayerNorm


def _test_quantize_layernorm(batch_size, tokens, embeddings, dtype, activations, device):
    # Instantiate a normalization layer
    norm = torch.nn.LayerNorm(embeddings).to(dtype).to(device)
    qnorm = QLayerNorm.from_module(norm, activations=activations)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), qtype=activations, dtype=dtype).to(device)
    # Calibrate to avoid clipping and to set the correct dtype
    with torch.no_grad(), Calibration():
        qout = qnorm(qinputs)
    qout = qnorm(qinputs)
    assert isinstance(qout, QTensor)
    assert qout.dtype == dtype
    assert qout.qtype == activations
    # Compare with the float results
    out = norm(qinputs.dequantize())
    # We need to increase atol for float16 dtype
    dtype_atol = {torch.float32: 1e-4, torch.float16: 1e-3}[dtype]
    # We also need to increase atol for float8 qtypes
    atol = {qint8: dtype_atol, qfloat8_e5m2: 5e-3, qfloat8_e4m3fn: 5e-3}[activations]
    assert_similar(out, qout, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.skip_device("cpu")
def test_quantize_layernorm_float16_activations_int8(batch_size, tokens, embeddings, device):
    _test_quantize_layernorm(batch_size, tokens, embeddings, torch.float16, qint8, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
def test_quantize_layernorm_float32_activations_int8(batch_size, tokens, embeddings, device):
    _test_quantize_layernorm(batch_size, tokens, embeddings, torch.float32, qint8, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn],
    ids=["a-float8-e5m2", "a-float8-e4m3"],
)
@pytest.mark.skip_device("cpu")
@pytest.mark.skip_device("mps")
def test_quantize_layernorm_float16_activations_float8(batch_size, tokens, embeddings, activations, device):
    _test_quantize_layernorm(batch_size, tokens, embeddings, torch.float16, activations, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn],
    ids=["a-float8-e5m2", "a-float8-e4m3"],
)
@pytest.mark.skip_device("cpu")
@pytest.mark.skip_device("mps")
def test_quantize_layernorm_float32_activations_float8(batch_size, tokens, embeddings, activations, device):
    _test_quantize_layernorm(batch_size, tokens, embeddings, torch.float32, activations, device)


def test_quantize_layernom_no_activation():
    norm = torch.nn.LayerNorm(32)
    qnorm = QLayerNorm.from_module(norm, activations=None)
    assert qnorm is None
