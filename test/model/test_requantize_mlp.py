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

import pytest
import torch
from helpers import get_device_memory, random_tensor
from test_quantize_mlp import MLP, save_and_reload_state_dict

from optimum.quanto import (
    Calibration,
    freeze,
    qint8,
    quantize,
    requantize,
)
from optimum.quanto.nn import QModuleMixin


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("serialization", ["weights_only", "pickle", "safetensors"])
def test_serialize_requantized_mlp(weights, dtype, serialization, device):
    input_features = 32
    hidden_features = 10
    output_features = 128
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    quantize(model, weights=weights)
    inputs = random_tensor((1, 10, input_features), dtype=dtype).to(device)
    with Calibration():
        model(inputs)
    freeze(model)
    model_reloaded = MLP(input_features, hidden_features, output_features).to(device)
    state_dict = save_and_reload_state_dict(model.state_dict(), serialization)
    requantize(model_reloaded, state_dict)
    for name, module in model.named_modules():
        if isinstance(module, QModuleMixin):
            module_reloaded = getattr(model_reloaded, name)
            assert module_reloaded.weight.qtype == module.weight.qtype
            assert module_reloaded.weight_qtype == module.weight_qtype
            assert module_reloaded.activation_qtype == module.activation_qtype
            assert torch.equal(module_reloaded.weight._data, module.weight._data)
            assert torch.equal(module_reloaded.weight._scale, module.weight._scale)
            assert torch.equal(module_reloaded.input_scale, module.input_scale)
            assert torch.equal(module_reloaded.output_scale, module.output_scale)


@pytest.mark.skip_device("cpu")
@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("weights_only", [True, False], ids=["weights-only", "pickle"])
@pytest.mark.parametrize("serialization", ["weights_only", "pickle", "safetensors"])
def test_requantized_mlp_device_memory(weights, dtype, weights_only, device, serialization):
    # We might not start from a clean state
    input_features = 1024
    hidden_features = 2048
    output_features = 1024
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    full_precision_memory = get_device_memory(device)
    quantize(model, weights=weights)
    freeze(model)
    quantized_memory = get_device_memory(device)
    assert quantized_memory < full_precision_memory
    state_dict = save_and_reload_state_dict(model.state_dict(), serialization)
    # Free device memory
    del model
    reloaded_model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    requantize(reloaded_model, state_dict)
    # Free device memory
    del state_dict
    requantized_memory = get_device_memory(device)
    assert requantized_memory <= quantized_memory
