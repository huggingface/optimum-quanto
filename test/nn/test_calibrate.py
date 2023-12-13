import pytest
import torch
from helpers import random_qtensor

from quanto.quantization import QTensor, calibration
from quanto.quantization.nn import QLinear


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("per_axis", [True, False], ids=["per-axis", "per-tensor"])
def test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, per_axis, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(device)
    qlinear = QLinear.from_module(linear, activations=torch.int8)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    # Run a first inference without calibration
    with torch.no_grad():
        qout = qlinear(qinputs)
    assert torch.all(qlinear.input_scale == 1)
    assert torch.all(qlinear.output_scale == 1)
    # Calibrate to adjust input and output scales and set the correct dtype
    with torch.no_grad(), calibration(per_axis=per_axis):
        qout = qlinear(qinputs)
    assert torch.any(qlinear.input_scale != 1)
    assert torch.any(qlinear.output_scale != 1)
    if per_axis:
        assert qout.axis == 2


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
    model.linear1 = QLinear.from_module(model.linear1, activations=torch.int8)
    model.linear2 = QLinear.from_module(model.linear2, activations=torch.int8)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32)
    with torch.no_grad(), calibration(per_axis=per_axis):
        qout = model(qinputs)
    assert torch.any(model.linear1.input_scale != 1)
    assert torch.any(model.linear1.output_scale != 1)
    assert torch.any(model.linear2.input_scale != 1)
    assert torch.any(model.linear2.output_scale != 1)
    if not per_axis:
        assert isinstance(qout, QTensor)
        assert qout.itype == torch.int8
