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

from contextlib import nullcontext

import pytest
import torch
from helpers import assert_similar, get_device_memory, random_tensor

from optimum.quanto import (
    AbsmaxOptimizer,
    ActivationQBytesTensor,
    Calibration,
    MaxOptimizer,
    QLinear,
    QTensor,
    absmax_scale,
    freeze,
    qfloat8_e4m3fn,
    qfloat8_e4m3fnuz,
    qfloat8_e5m2,
    qint4,
    qint8,
    quantize,
    quantize_activation,
)


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


def _test_quantize_mlp(weights, activations, optimizer, frozen, device, atol=1e-6):
    model = MLP(32, 10, 128).to(device)
    inputs = random_tensor((1, 32), dtype=torch.float32, device=device)
    output = model(inputs)
    quantize(model, weights=weights, activations=activations, optimizer=optimizer)
    if frozen:
        freeze(model)
    check_mlp(model, frozen)
    if activations is not None:
        inputs = quantize_activation(inputs, qtype=activations, scale=absmax_scale(inputs))
        context = Calibration
    else:
        context = nullcontext
    with context():
        qoutput = model(inputs)
    if activations is not None:
        assert isinstance(qoutput, ActivationQBytesTensor)
    assert_similar(output, qoutput, atol=atol)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
def test_quantize_mlp_weights_only(weights, frozen, device):
    _test_quantize_mlp(weights, None, None, frozen, device)


@pytest.mark.skip_device("mps")
@pytest.mark.parametrize("weights", [qfloat8_e4m3fn], ids=["w-float8_e4m3fn"])
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
def test_quantize_mlp_weights_only_float8(weights, frozen, device):
    _test_quantize_mlp(weights, None, None, frozen, device)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
@pytest.mark.skip_device("mps")
def test_quantize_mlp_int8_activations(weights, frozen, device):
    _test_quantize_mlp(weights, qint8, None, frozen, device, atol=1e-3)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn, qfloat8_e4m3fnuz],
    ids=["a-qfloat8-e5m2", "a-qfloat8-e4m3", "a-float8-e4m3-uz"],
)
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
@pytest.mark.skip_device("mps")
def test_quantize_mlp_float8_activations(weights, activations, frozen, device):
    atol = {qfloat8_e4m3fn: 1e-3, qfloat8_e4m3fnuz: 1e-3, qfloat8_e5m2: 1e-2}[activations]
    _test_quantize_mlp(weights, activations, None, frozen, device, atol=atol)


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


@pytest.mark.parametrize(
    "weights, optimizer", [[qint8, AbsmaxOptimizer()], [qint4, MaxOptimizer()]], ids=["w-qint8", "w-qint4"]
)
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "non-frozen"])
def test_quantize_mlp_weights_only_optimizers(weights, optimizer, frozen, device):
    atol = {qint4: 1e-4, qint8: 1e-6}[weights]
    _test_quantize_mlp(weights, None, optimizer, frozen, device, atol=atol)


@pytest.mark.parametrize(
    "weights, optimizer", [[qint8, MaxOptimizer()], [qint4, AbsmaxOptimizer()]], ids=["w-qint8", "w-qint4"]
)
def test_quantize_mlp_wrong_optimizer(weights, optimizer, device):
    with pytest.raises(ValueError):
        _test_quantize_mlp(weights, None, optimizer, False, device)
