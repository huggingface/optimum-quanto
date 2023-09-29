import os
from tempfile import TemporaryDirectory

import pytest
import torch
from helpers import q_assert_close, random_qtensor

from quanto.quantization.calibrate import calibration
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
    assert torch.allclose(qout.dequantize(), int_qout._data)


def test_qlinear_serialization():
    tokens = 10
    embeddings = 32
    linear = torch.nn.Linear(embeddings, embeddings)
    qlinear = QLinear.from_module(linear)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), calibration():
        qlinear(qinputs)
    with TemporaryDirectory() as tmpdir:
        qlinear_file = os.path.join(tmpdir, "qlinear.pt")
        torch.save(qlinear.state_dict(), qlinear_file)
        qlinear_reloaded = QLinear(embeddings, embeddings)
        # When reloading we must assign instead of copying to force quantized tensors assignment
        qlinear_reloaded.load_state_dict(torch.load(qlinear_file), assign=True)
    for attr in ["weight", "bias"]:
        t = getattr(qlinear, attr)
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
    assert torch.equal(qout._data, int_qout._data)
    assert torch.equal(qout._scale, int_qout._scale)
