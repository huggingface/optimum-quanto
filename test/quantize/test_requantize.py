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
from helpers import get_device_memory, random_tensor
from safetensors.torch import load_file, save_file
from test_quantize_mlp import MLP

from optimum.quanto import Calibration, freeze, qint4, qint8, quantization_map, quantize, requantize
from optimum.quanto.nn import QModuleMixin


def save_and_reload_state_dict(state_dict, serialization):
    if serialization == "safetensors":
        with NamedTemporaryFile() as tmp_file:
            save_file(state_dict, tmp_file.name)
            return load_file(tmp_file.name)
    else:
        b = io.BytesIO()
        torch.save(state_dict, b)
        b.seek(0)
        weights_only = serialization == "weights_only"
        return torch.load(b, weights_only=weights_only)


@pytest.mark.parametrize(
    "input_features, hidden_features, output_features",
    [(32, 10, 128), (1024, 1024, 1024)],
    ids=["small", "large"],
)
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("serialization", ["weights_only", "pickle", "safetensors"])
@pytest.mark.parametrize("activations", [None, qint8], ids=["a-none", "a-qint8"])
def test_requantize_serialized_model(
    input_features, hidden_features, output_features, weights, activations, dtype, serialization, device
):
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    quantize(model, weights=weights, activations=activations)
    inputs = random_tensor((1, 10, input_features), dtype=dtype).to(device)
    if activations is not None:
        with Calibration():
            model(inputs)
    freeze(model)
    qmap = quantization_map(model)
    model_reloaded = MLP(input_features, hidden_features, output_features).to(device)
    state_dict = save_and_reload_state_dict(model.state_dict(), serialization)
    requantize(model_reloaded, state_dict, qmap)
    for name, module in model.named_modules():
        if isinstance(module, QModuleMixin):
            module_reloaded = getattr(model_reloaded, name)
            assert torch.equal(module_reloaded.weight, module.weight)
            assert module_reloaded.weight_qtype == module.weight_qtype
            assert module_reloaded.activation_qtype == module.activation_qtype
            assert torch.equal(module_reloaded.input_scale, module.input_scale)
            assert torch.equal(module_reloaded.output_scale, module.output_scale)


@pytest.mark.skip_device("cpu")
@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("serialization", ["weights_only", "pickle", "safetensors"])
def test_requantized_model_device_memory(weights, dtype, serialization, device):
    input_features = 1024
    hidden_features = 2048
    output_features = 1024
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    full_precision_memory = get_device_memory(device)
    quantize(model, weights=weights)
    freeze(model)
    qmap = quantization_map(model)
    quantized_memory = get_device_memory(device)
    assert quantized_memory < full_precision_memory
    state_dict = save_and_reload_state_dict(model.state_dict(), serialization)
    # Free device memory
    del model
    with torch.device("meta"):
        reloaded_model = MLP(input_features, hidden_features, output_features).to(dtype)
    requantize(reloaded_model, state_dict, qmap, device)
    # Free device memory
    del state_dict
    requantized_memory = get_device_memory(device)
    assert requantized_memory <= quantized_memory
