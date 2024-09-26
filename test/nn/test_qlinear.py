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
from contextlib import nullcontext

import pytest
import torch
from helpers import assert_similar, random_qactivation, random_tensor

from optimum.quanto import (
    ActivationQBytesTensor,
    Calibration,
    absmax_scale,
    qfloat8,
    qfloat8_e4m3fn,
    qfloat8_e4m3fnuz,
    qfloat8_e5m2,
    qint4,
    qint8,
    quantize_activation,
)
from optimum.quanto.nn import QLinear


def _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, activations, dtype, device, atol=None):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(dtype).to(device)
    qlinear = QLinear.from_module(linear, weights=weights, activations=activations)
    assert qlinear.qweight.qtype == weights
    input_shape = (batch_size, tokens, embeddings)
    if activations is not None:
        qinputs = random_qactivation(input_shape, qtype=activations, dtype=dtype).to(device)
        inputs = qinputs.dequantize()
    else:
        inputs = random_tensor(input_shape, dtype=dtype, device=device)
    # Run an inference with Calibration to get the correct output dtype
    context = nullcontext if activations is None else Calibration
    with torch.no_grad(), context():
        qout = qlinear(inputs if activations is None else qinputs)
    if activations is not None:
        assert isinstance(qout, ActivationQBytesTensor)
        assert qout.qtype == activations
    # Align linear weights with quantized linear weights for comparison
    linear.weight = torch.nn.Parameter(qlinear.qweight.dequantize())
    out = linear(inputs)
    assert_similar(out, qout, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(10, 32), (10, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
def test_quantize_linear_float16_activations_int8(batch_size, tokens, embeddings, use_bias, dtype, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, qint8, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(10, 32), (10, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
def test_quantize_linear_float32_activations_int8(batch_size, tokens, embeddings, use_bias, weights, device):
    # Default atol for float32 is 1e-6
    atol = 1e-4
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, qint8, torch.float32, device, atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(10, 32), (10, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e4m3fn, qfloat8_e4m3fnuz],
    ids=["a-qfloat8-e4m3", "a-float8-e4m3-uz"],
)
@pytest.mark.skip_device("mps")
def test_quantize_linear_float16_activations_float8(
    batch_size, tokens, embeddings, use_bias, dtype, weights, activations, device
):
    atol = 5e-3
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, activations, dtype, device, atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn, qfloat8_e4m3fnuz],
    ids=["a-qfloat8-e5m2", "a-qfloat8-e4m3", "a-float8-e4m3-uz"],
)
@pytest.mark.skip_device("mps")
def test_quantize_linear_float32_activations_float8(
    batch_size, tokens, embeddings, use_bias, weights, activations, device
):
    atol = 5e-3
    _test_quantize_linear(
        batch_size, tokens, embeddings, use_bias, weights, activations, torch.float32, device, atol=atol
    )


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(10, 32), (10, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8, qfloat8], ids=["w-qint4", "w-qint8", "float8"])
def test_quantize_linear_float16_weight_only(batch_size, tokens, embeddings, use_bias, weights, device):
    if device.type == "mps" and weights == qfloat8:
        pytest.skip("Float 8 are not supported on MPS device")
    atol = None
    if device.type == "cuda" and weights == qfloat8 and embeddings % 64 == 0:
        # FIXME: accuracy is slightly worse using MARLIN FP8 kernels
        atol = 1e-2
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, None, torch.float16, device, atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(10, 32), (10, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-qint4", "w-qint8"])
def test_quantize_linear_float32_weight_only(batch_size, tokens, embeddings, use_bias, weights, device):
    _test_quantize_linear(batch_size, tokens, embeddings, use_bias, weights, None, torch.float32, device)


@pytest.mark.parametrize("tokens, embeddings", [(10, 32), (10, 256)])
@pytest.mark.parametrize("activations", [None, qint8, qfloat8], ids=["a-float", "a-qint8", "a-float8"])
@pytest.mark.parametrize("weights", [qint4, qint8, qfloat8], ids=["w-qint4", "w-qint8", "w-float8"])
def test_qlinear_gradient(tokens, embeddings, activations, weights, device):
    if device.type == "mps" and (activations == qfloat8 or weights == qfloat8):
        pytest.skip("Float8 is not supported on MPS device")
    batch_size = 10
    linear = torch.nn.Linear(embeddings, embeddings).to(device)
    qlinear = QLinear.from_module(linear, weights=weights, activations=activations)
    assert qlinear.weight.requires_grad is True
    assert qlinear.bias.requires_grad is True
    # Run an inference with dynamically quantized inputs
    inputs = random_tensor((batch_size, tokens, embeddings), dtype=torch.float32, device=device)
    inputs.requires_grad = True
    if activations is None:
        qout = qlinear(inputs)
        float_inputs = inputs.clone().detach()
    else:
        qinputs = quantize_activation(inputs, qtype=activations, scale=absmax_scale(inputs, activations))
        qout = qlinear(qinputs)
        # Run an equivalent inference with float inputs
        float_inputs = qinputs.dequantize().clone().detach()
    float_inputs.requires_grad = True
    out = linear(float_inputs)
    # Outputs are not identical because of the quantization
    assert not torch.equal(qout, out)
    # Compute gradients and compare
    gradient = torch.randn(qout.size()).to(device)
    qout.backward(gradient)
    out.backward(gradient)
    # Bias gradients are identical because they don't depend on inputs and weights
    atol = 1e-6
    assert_similar(qlinear.bias.grad, linear.bias.grad, atol=atol)
    # Weights gradients are nearly identical, based on identical inputs through subtly different graphs
    atol = 1e-5
    assert_similar(qlinear.weight.grad, linear.weight.grad, atol=atol)
    # Inputs gradients are slightly different because they depend on the quantized weights
    atol = {qint8: 1e-5, qint4: 5e-3, qfloat8: 5e-3}[weights]
    assert_similar(inputs.grad, float_inputs.grad, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8, qfloat8], ids=["w-int4", "w-int8", "w-float8"])
def test_move_qlinear(dtype, use_bias, weights, device):
    linear = torch.nn.Linear(1024, 1024, bias=use_bias).to(dtype)
    qlinear = QLinear.from_module(linear, weights=weights)
    qlinear.freeze()
    qlinear.to(device)
    inner_tensor_names, _ = qlinear.weight.__tensor_flatten__()
    for name in inner_tensor_names:
        assert getattr(qlinear.weight, name).device.type == device.type
    if use_bias:
        assert qlinear.bias.device.type == device.type


@pytest.mark.parametrize("features", [10, 256], ids=["per-axis", "per-group"])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8, qfloat8], ids=["w-qint4", "w-qint8", "w-qfloat8"])
@pytest.mark.parametrize("activations", [None, qint8, qfloat8], ids=["a-float", "a-qint8", "a-qfloat8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("weights_only", [True, False], ids=["weights-only", "pickle"])
def test_qlinear_serialization(features, use_bias, activations, weights, dtype, weights_only, device):
    if device.type == "mps" and (activations == qfloat8 or weights == qfloat8):
        pytest.skip("Float8 is not supported on MPS device")
    linear = torch.nn.Linear(features, features, bias=use_bias).to(dtype).to(device)
    qlinear = QLinear.from_module(linear, weights=weights, activations=activations)
    if activations is not None:
        qinputs = random_qactivation((10, 10, features), qtype=activations, dtype=dtype).to(device)
        with Calibration():
            qlinear(qinputs)
    qlinear.freeze()
    b = io.BytesIO()
    torch.save(qlinear.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b, weights_only=weights_only)
    qlinear_reloaded = QLinear(features, features, weights=weights, activations=activations, bias=use_bias).to(device)
    qlinear_reloaded.load_state_dict(state_dict)
    assert qlinear_reloaded.weight_qtype == weights
    w = qlinear.weight
    w_reloaded = qlinear_reloaded.weight
    assert torch.equal(w, w_reloaded)
    if activations is not None:
        assert qlinear_reloaded.activation_qtype == activations
        for attr in ["input_scale", "output_scale"]:
            v = getattr(qlinear, attr)
            v_reloaded = getattr(qlinear_reloaded, attr)
            assert torch.equal(v, v_reloaded)
