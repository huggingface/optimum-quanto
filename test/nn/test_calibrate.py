import pytest
import torch
from helpers import q_assert_close, random_qtensor

from quanto.quantization import calibration, freeze
from quanto.quantization.nn import QLinear


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("per_axis", [True, False], ids=["per-axis", "per-tensor"])
def test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, per_axis, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(device)
    qlinear = QLinear.from_module(linear)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    # Run a first inference without calibration
    with torch.no_grad():
        qout = qlinear(qinputs)
    assert qout._data.dtype == torch.int8
    assert torch.all(qout._scale == 1.0)
    assert torch.all(qlinear.in_scale == 1.0)
    assert torch.all(qlinear.out_scale == 1.0)
    # Calibrate to adjust input and output scales
    with torch.no_grad(), calibration(per_axis=per_axis):
        qout = qlinear(qinputs)
    assert qout._data.dtype == torch.int8
    assert torch.all(qout._scale != 1.0)
    assert torch.all(qlinear.in_scale != 1.0)
    assert torch.all(qlinear.out_scale != 1.0)
    if per_axis:
        assert qout.axis == 2
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
    assert torch.equal(qout._scale, int_qout._scale)
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1


@pytest.mark.parametrize("per_axis", [True, False], ids=["per-axis", "per-tensor"])
def test_calibrate_custom_module(per_axis):
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
    with torch.no_grad(), calibration(per_axis=per_axis):
        qout = model(qinputs)
    assert torch.all(model.linear1.in_scale != 1)
    assert torch.all(model.linear1.out_scale != 1)
    assert torch.all(model.linear2.in_scale != 1)
    assert torch.all(model.linear2.out_scale != 1)
    assert torch.all(qout._scale != 1)
    if per_axis:
        assert qout.axis == 2
    with torch.no_grad():
        int_qout = model(qinputs)
    assert torch.equal(qout._scale, int_qout._scale)
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1
