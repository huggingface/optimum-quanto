import pytest
import torch
from helpers import random_qtensor

from quanto.quantization import QLinear, QuantizedTensor, freeze, quantize


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.mid_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = torch.nn.functional.relu(self.input_layer(inputs))
        x = torch.nn.functional.relu(self.mid_layer(x))
        return torch.nn.functional.softmax(self.output_layer(x))


def check_mlp(model, frozen):
    assert isinstance(model.input_layer, QLinear)
    assert isinstance(model.mid_layer, QLinear)
    assert isinstance(model.output_layer, QLinear)
    if frozen:
        assert isinstance(model.input_layer.weight, QuantizedTensor)
        assert isinstance(model.mid_layer.weight, QuantizedTensor)
        assert isinstance(model.output_layer.weight, QuantizedTensor)


def check_outputs(model, batch_size, input_features, device):
    qinputs = random_qtensor((batch_size, input_features), dtype=torch.float32).to(device)
    qout = model(qinputs)
    assert isinstance(qout, QuantizedTensor)


@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
def test_quantize_mlp(frozen, device):
    model = MLP(32, 10, 128).to(device)
    quantize(model)
    if frozen:
        freeze(model)
    check_mlp(model, frozen)
    check_outputs(model, 1, 32, device)
