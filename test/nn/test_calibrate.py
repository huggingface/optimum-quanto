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
from helpers import random_qactivation

from optimum.quanto import Calibration, qfloat8_e4m3fn, qfloat8_e4m3fnuz, qfloat8_e5m2, qint8
from optimum.quanto.nn import QLinear


def _test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, activations, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(device)
    qlinear = QLinear.from_module(linear, weights=qint8, activations=activations)
    qinputs = random_qactivation(
        (batch_size, tokens, embeddings), qtype=activations, dtype=torch.float32, device=device
    )
    # Run a first inference without Calibration
    with torch.no_grad():
        qout = qlinear(qinputs)
    assert torch.all(qlinear.input_scale == 1)
    assert torch.all(qlinear.output_scale == 1)
    # Calibrate to adjust input and output scales and set the correct dtype
    with torch.no_grad(), Calibration():
        qout = qlinear(qinputs)
    assert qout.qtype == activations
    assert torch.any(qlinear.input_scale != 1)
    assert torch.any(qlinear.output_scale != 1)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_calibrate_qlinear_activations_int8(batch_size, tokens, embeddings, use_bias, device):
    _test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, qint8, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn, qfloat8_e4m3fnuz],
    ids=["a-qfloat8-e5m2", "a-qfloat8-e4m3", "a-qfloat8-e4m3-uz"],
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
    model.linear1 = QLinear.from_module(model.linear1, weights=qint8, activations=activations)
    model.linear2 = QLinear.from_module(model.linear2, weights=qint8, activations=activations)
    qinputs = random_qactivation((1, tokens, embeddings), qtype=activations, dtype=torch.float32, device=device)
    with torch.no_grad(), Calibration():
        qout = model(qinputs)
    assert torch.any(model.linear1.input_scale != 1)
    assert torch.any(model.linear1.output_scale != 1)
    assert torch.any(model.linear2.input_scale != 1)
    assert torch.any(model.linear2.output_scale != 1)
    assert qout.qtype == activations


def test_calibrate_custom_module_activations_int8(device):
    _test_calibrate_custom_module(qint8, device)


@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn, qfloat8_e4m3fnuz],
    ids=["a-qfloat8-e5m2", "a-qfloat8-e4m3", "a-qfloat8-e4m3-uz"],
)
@pytest.mark.skip_device("mps")
def test_calibrate_custom_module_activations_float8(activations, device):
    _test_calibrate_custom_module(activations, device)
