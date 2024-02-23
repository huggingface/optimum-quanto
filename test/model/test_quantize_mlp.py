import io

import pytest
import torch
from helpers import assert_similar, random_qtensor, random_tensor

from quanto import Calibration, QLinear, QTensor, freeze, qfloat8_e4m3fn, qfloat8_e5m2, qint8, quantize
from quanto.nn import QModuleMixin


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


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_serialize_quantized_mlp(weights, dtype, device):
    if dtype == torch.float16 and device.type == "cpu":
        pytest.skip("Matrix multiplication is not supported for float16 on CPU")
    input_features = 32
    hidden_features = 10
    output_features = 128
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    quantize(model, weights=weights)
    inputs = random_tensor((1, 10, input_features), dtype=dtype).to(device)
    with Calibration():
        model(inputs)
    freeze(model)
    b = io.BytesIO()
    torch.save(model.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b)
    model_reloaded = MLP(input_features, hidden_features, output_features).to(device)
    quantize(model_reloaded)
    # When reloading we must assign instead of copying to force quantized tensors assignment
    model_reloaded.load_state_dict(state_dict, assign=True)
    for name, module in model.named_modules():
        if isinstance(module, QModuleMixin):
            module_reloaded = getattr(model_reloaded, name)
            assert torch.equal(module_reloaded.weight._data, module.weight._data)
            assert torch.equal(module_reloaded.weight._scale, module.weight._scale)
            assert torch.equal(module_reloaded.input_scale, module.input_scale)
            assert torch.equal(module_reloaded.output_scale, module.output_scale)
