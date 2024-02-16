import pytest
import torch
from helpers import assert_similar, random_qtensor

from quanto import Calibration, QLinear, QTensor, freeze, qfloat8_e4m3fn, qfloat8_e5m2, qint8, quantize


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.mid_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = torch.nn.functional.relu(self.input_layer(inputs))
        x = torch.nn.functional.relu(self.mid_layer(x))
        return torch.nn.functional.softmax(self.output_layer(x), dim=-1)


def check_mlp(model, frozen):
    assert isinstance(model.input_layer, QLinear)
    assert isinstance(model.mid_layer, QLinear)
    assert isinstance(model.output_layer, QLinear)
    if frozen:
        assert isinstance(model.input_layer.weight, QTensor)
        assert isinstance(model.mid_layer.weight, QTensor)
        assert isinstance(model.output_layer.weight, QTensor)


def get_outputs(model, batch_size, input_features, device):
    qinputs = random_qtensor((batch_size, input_features), dtype=torch.float32).to(device)
    return model(qinputs)


def _test_quantize_mlp(weights, activations, frozen, device):
    model = MLP(32, 10, 128).to(device)
    output = get_outputs(model, 1, 32, device)
    quantize(model, weights=weights, activations=activations)
    if frozen:
        freeze(model)
    check_mlp(model, frozen)
    with Calibration():
        qoutput = get_outputs(model, 1, 32, device)
    if activations is not None:
        assert isinstance(qoutput, QTensor)
    # Don't expect more than a 0.99 similarity
    assert_similar(output, qoutput, atol=1e-2)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
def test_quantize_mlp_weights_only(weights, frozen, device):
    _test_quantize_mlp(weights, None, frozen, device)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
@pytest.mark.skip_device("mps")
def test_quantize_mlp_int8_activations(weights, frozen, device):
    _test_quantize_mlp(weights, qint8, frozen, device)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize(
    "activations",
    [None, qint8, qfloat8_e5m2, qfloat8_e4m3fn],
    ids=["a-float", "a-qint8", "a-qfloat8-e5m2", "a-qfloat8-e4m3"],
)
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
@pytest.mark.skip_device("mps")
def test_quantize_mlp_float8_activations(weights, activations, frozen, device):
    _test_quantize_mlp(weights, activations, frozen, device)
