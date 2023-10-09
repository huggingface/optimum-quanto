import os
from tempfile import TemporaryDirectory

import pytest
import torch
from helpers import q_assert_close, random_qtensor

from quanto.quantization import calibration, freeze
from quanto.quantization.nn import QLinear


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_quantize_linear(batch_size, tokens, embeddings, use_bias, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(device)
    qlinear = QLinear.from_module(linear)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), calibration():
        qout = qlinear(qinputs)
    # Freeze to set quantized weights
    freeze(qlinear)
    # Align linear weights with quantized linear weights for comparison
    linear.weight = torch.nn.Parameter(qlinear.weight.dequantize())
    if use_bias:
        linear.bias = torch.nn.Parameter(qlinear.bias.dequantize())
    out = linear(qinputs.dequantize())
    q_assert_close(out, qout)
    # Now run an inference without calibrating
    with torch.no_grad():
        int_qout = qlinear(qinputs)
    assert qout._scale == int_qout._scale
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1


@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_qlinear_serialization(use_bias):
    tokens = 10
    embeddings = 32
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias)
    qlinear = QLinear.from_module(linear)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), calibration():
        qlinear(qinputs)
    # Freeze linear to store quantized weights and biases
    qlinear.freeze()
    with TemporaryDirectory() as tmpdir:
        qlinear_file = os.path.join(tmpdir, "qlinear.pt")
        torch.save(qlinear.state_dict(), qlinear_file)
        qlinear_reloaded = QLinear(embeddings, embeddings, bias=use_bias)
        # When reloading we must assign instead of copying to force quantized tensors assignment
        qlinear_reloaded.load_state_dict(torch.load(qlinear_file), assign=True)
    for attr in ["weight", "bias"]:
        t = getattr(qlinear, attr)
        if t is not None:
            t_reloaded = getattr(qlinear_reloaded, attr)
            assert torch.equal(t._data, t_reloaded._data)
            assert torch.equal(t._scale, t_reloaded._scale)
    for attr in ["in_scale", "out_scale"]:
        v = getattr(qlinear, attr)
        v_reloaded = getattr(qlinear_reloaded, attr)
        assert torch.equal(v, v_reloaded)


def test_quantize_custom_module():
    tokens = 10
    embeddings = 32

    class TwoLinearModel(torch.nn.Module):
        def __init__(self, embeddings):
            super().__init__()
            self.linear1 = torch.nn.Linear(embeddings, embeddings)
            self.linear2 = torch.nn.Linear(embeddings, embeddings)

        def forward(self, input):
            return self.linear2(self.linear1(input))

    model = TwoLinearModel(embeddings)
    model.linear1 = QLinear.from_module(model.linear1)
    model.linear2 = QLinear.from_module(model.linear2)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32)
    with torch.no_grad(), calibration():
        qout = model(qinputs)
    assert model.linear1.in_scale != 1
    assert model.linear1.out_scale != 1
    assert model.linear2.in_scale != 1
    assert model.linear2.out_scale != 1
    with torch.no_grad():
        int_qout = model(qinputs)
    assert torch.equal(qout._scale, int_qout._scale)
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1


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
