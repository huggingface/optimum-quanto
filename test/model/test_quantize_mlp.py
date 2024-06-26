# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from tempfile import NamedTemporaryFile

import pytest
import torch
from helpers import assert_similar, get_device_memory, random_qactivation, random_tensor

from optimum.quanto import (
    AbsmaxOptimizer,
    Calibration,
    MaxOptimizer,
    QBytesTensor,
    QLinear,
    QTensor,
    freeze,
    qfloat8_e4m3fn,
    qfloat8_e5m2,
    qint4,
    qint8,
    quantize,
    safe_load,
    safe_save,
)
from optimum.quanto.nn import QModuleMixin


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
    qinputs = random_qactivation((batch_size, input_features), dtype=torch.float32).to(device)
    return model(qinputs)


def _test_quantize_mlp(weights, activations, optimizer, frozen, device):
    model = MLP(32, 10, 128).to(device)
    output = get_outputs(model, 1, 32, device)
    quantize(model, weights=weights, activations=activations, optimizer=optimizer)
    if frozen:
        freeze(model)
    check_mlp(model, frozen)
    with Calibration():
        qoutput = get_outputs(model, 1, 32, device)
    if activations is not None:
        assert isinstance(qoutput, QBytesTensor)
    # Don't expect more than a 0.99 similarity
    assert_similar(output, qoutput, atol=1e-2)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
def test_quantize_mlp_weights_only(weights, frozen, device):
    _test_quantize_mlp(weights, None, None, frozen, device)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
@pytest.mark.skip_device("mps")
def test_quantize_mlp_int8_activations(weights, frozen, device):
    _test_quantize_mlp(weights, qint8, None, frozen, device)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize(
    "activations",
    [None, qint8, qfloat8_e5m2, qfloat8_e4m3fn],
    ids=["a-float", "a-qint8", "a-qfloat8-e5m2", "a-qfloat8-e4m3"],
)
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
@pytest.mark.skip_device("mps")
def test_quantize_mlp_float8_activations(weights, activations, frozen, device):
    _test_quantize_mlp(weights, activations, None, frozen, device)


def save_and_reload_state_dict(state_dict, serialization):
    if serialization == "safetensors":
        with NamedTemporaryFile() as tmp_file:
            safe_save(state_dict, tmp_file.name)
            return safe_load(tmp_file.name)
    else:
        b = io.BytesIO()
        torch.save(state_dict, b)
        b.seek(0)
        weights_only = serialization == "weights_only"
        return torch.load(b, weights_only=weights_only)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("serialization", ["weights_only", "pickle", "safetensors"])
def test_serialize_quantized_mlp(weights, dtype, serialization, device):
    input_features = 32
    hidden_features = 10
    output_features = 128
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    quantize(model, weights=weights)
    inputs = random_tensor((1, 10, input_features), dtype=dtype).to(device)
    with Calibration():
        model(inputs)
    freeze(model)
    state_dict = save_and_reload_state_dict(model.state_dict(), serialization)
    model_reloaded = MLP(input_features, hidden_features, output_features).to(device)
    quantize(model_reloaded)
    model_reloaded.load_state_dict(state_dict)
    for name, module in model.named_modules():
        if isinstance(module, QModuleMixin):
            module_reloaded = getattr(model_reloaded, name)
            assert torch.equal(module_reloaded.weight._data, module.weight._data)
            assert torch.equal(module_reloaded.weight._scale, module.weight._scale)
            assert torch.equal(module_reloaded.input_scale, module.input_scale)
            assert torch.equal(module_reloaded.output_scale, module.output_scale)


@pytest.mark.skip_device("cpu")
@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("weights_only", [True, False], ids=["weights-only", "pickle"])
def test_quantized_mlp_device_memory(weights, dtype, weights_only, device):
    # We might not start from a clean state
    base_memory = get_device_memory(device)
    input_features = 1024
    hidden_features = 2048
    output_features = 1024
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    full_precision_memory = get_device_memory(device)
    assert full_precision_memory > base_memory
    quantize(model, weights=weights)
    freeze(model)
    quantized_memory = get_device_memory(device)
    assert quantized_memory > base_memory
    assert quantized_memory < full_precision_memory
    # Serialize model
    b = io.BytesIO()
    torch.save(model.state_dict(), b)
    # Free device memory
    del model
    assert get_device_memory(device) == base_memory
    # Reload state dict on CPU
    b.seek(0)
    state_dict = torch.load(b, map_location=torch.device("cpu"), weights_only=weights_only)
    assert get_device_memory(device) == base_memory
    # Create an empty model and quantize it with the same parameters
    with torch.device("meta"):
        model_reloaded = MLP(input_features, hidden_features, output_features)
        assert get_device_memory(device) == base_memory
        quantize(model_reloaded)
        assert get_device_memory(device) == base_memory
    # Reload the state dict, still on CPU
    model_reloaded.load_state_dict(state_dict, assign=True)
    assert get_device_memory(device) == base_memory
    # Finally, move the model to the device
    model_reloaded.to(device)
    reloaded_memory = get_device_memory(device)
    # Device memory can be lower when reloading (less fragmentation ?)
    assert reloaded_memory <= quantized_memory


@pytest.mark.parametrize(
    "weights, optimizer", [[qint8, AbsmaxOptimizer()], [qint4, MaxOptimizer()]], ids=["w-qint8", "w-qint4"]
)
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
def test_quantize_mlp_weights_only_optimizers(weights, optimizer, frozen, device):
    _test_quantize_mlp(weights, None, optimizer, frozen, device)


@pytest.mark.parametrize(
    "weights, optimizer", [[qint8, MaxOptimizer()], [qint4, AbsmaxOptimizer()]], ids=["w-qint8", "w-qint4"]
)
def test_quantize_mlp_wrong_optimizer(weights, optimizer, device):
    with pytest.raises(ValueError):
        _test_quantize_mlp(weights, None, optimizer, False, device)
