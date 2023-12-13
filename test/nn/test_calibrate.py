import pytest
import torch
from helpers import random_qtensor

from quanto.quantization import calibration
from quanto.quantization.nn import QLinear


def _test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, activations, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(device)
    qlinear = QLinear.from_module(linear, activations=activations)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    # Run a first inference without calibration
    with torch.no_grad():
        qout = qlinear(qinputs)
    assert torch.all(qlinear.input_scale == 1)
    assert torch.all(qlinear.output_scale == 1)
    # Calibrate to adjust input and output scales and set the correct dtype
    with torch.no_grad(), calibration():
        qout = qlinear(qinputs)
    assert qout.itype == activations
    assert torch.any(qlinear.input_scale != 1)
    assert torch.any(qlinear.output_scale != 1)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_calibrate_qlinear_activations_int8(batch_size, tokens, embeddings, use_bias, device):
    _test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, torch.int8, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize(
    "activations",
    [torch.float8_e5m2, torch.float8_e4m3fn],
    ids=["a-float8-e5m2", "a-float8-e4m3"],
)
@pytest.mark.skip_device("mps")
def test_calibrate_qlinear_activations_float8(batch_size, tokens, embeddings, use_bias, activations, device):
    _test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, activations, device)


def _test_calibrate_custom_module(activations, device):
    tokens = 10
    embeddings = 32

    class TwoLinearModel(torch.nn.Module):
        def __init__(self, embeddings):
            super().__init__()
            self.linear1 = torch.nn.Linear(embeddings, embeddings)
            self.linear2 = torch.nn.Linear(embeddings, embeddings)

        def forward(self, input):
            return self.linear2(self.linear1(input))

    model = TwoLinearModel(embeddings).to(device)
    model.linear1 = QLinear.from_module(model.linear1, activations=activations)
    model.linear2 = QLinear.from_module(model.linear2, activations=activations)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32).to(device)
    with torch.no_grad(), calibration():
        qout = model(qinputs)
    assert torch.any(model.linear1.input_scale != 1)
    assert torch.any(model.linear1.output_scale != 1)
    assert torch.any(model.linear2.input_scale != 1)
    assert torch.any(model.linear2.output_scale != 1)
    assert qout.itype == activations


def test_calibrate_custom_module_activations_int8(device):
    _test_calibrate_custom_module(torch.int8, device)


@pytest.mark.parametrize(
    "activations",
    [torch.float8_e5m2, torch.float8_e4m3fn],
    ids=["a-float8-e5m2", "a-float8-e4m3"],
)
@pytest.mark.skip_device("mps")
def test_calibrate_custom_module_activations_float8(activations, device):
    _test_calibrate_custom_module(activations, device)
