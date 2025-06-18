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
from helpers import assert_similar, random_qactivation, random_tensor

from optimum.quanto import (
    ActivationQBytesTensor,
    Calibration,
    qfloat8_e4m3fn,
    qfloat8_e4m3fnuz,
    qfloat8_e5m2,
    qint4,
    qint8,
)
from optimum.quanto.nn import QConv2d


def _test_quantize_conv2d(batch_size, img_shape, out_channels, use_bias, weights, activations, dtype, device):
    conv2d = torch.nn.Conv2d(img_shape[0], out_channels, kernel_size=3, bias=use_bias).to(dtype).to(device)
    qconv2d = QConv2d.from_module(conv2d, weights=weights, activations=activations)
    assert qconv2d.qweight.qtype == weights
    inputs = random_tensor((batch_size,) + img_shape, dtype=dtype, device=device)
    # Run an inference with Calibration to get the correct output dtype
    with torch.no_grad(), Calibration():
        qout = qconv2d(inputs)
    if activations is not None:
        assert isinstance(qout, ActivationQBytesTensor)
        assert qout.qtype == activations
    # Align weights with quantized linear weights for comparison
    conv2d.weight = torch.nn.Parameter(qconv2d.qweight.dequantize())
    out = conv2d(inputs)
    # We need to increase atol for float16 dtype
    dtype_atol = {torch.float32: 1e-4, torch.float16: 1e-3}[dtype]
    # We also need to increase atol for float8 itypes
    atol = {None: dtype_atol, qint8: dtype_atol, qfloat8_e5m2: 5e-3, qfloat8_e4m3fn: 5e-3, qfloat8_e4m3fnuz: 5e-3}[
        activations
    ]
    assert_similar(out, qout, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("img_shape", [(3, 32, 32), (10, 32, 32)])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
def test_quantize_conv2d_float16_activations_int8(batch_size, img_shape, out_channels, use_bias, weights, device):
    _test_quantize_conv2d(batch_size, img_shape, out_channels, use_bias, weights, qint8, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("img_shape", [(3, 32, 32), (10, 32, 32)])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
def test_quantize_conv2d_float32_activations_int8(batch_size, img_shape, out_channels, use_bias, weights, device):
    _test_quantize_conv2d(batch_size, img_shape, out_channels, use_bias, weights, qint8, torch.float32, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("img_shape", [(3, 32, 32), (10, 32, 32)])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn, qfloat8_e4m3fnuz],
    ids=["a-float8-e5m2", "a-float8-e4m3", "a-float8_e4m3-uz"],
)
@pytest.mark.skip_device("mps")
def test_quantize_conv2d_float16_activations_float8(
    batch_size, img_shape, out_channels, use_bias, weights, activations, device
):
    _test_quantize_conv2d(batch_size, img_shape, out_channels, use_bias, weights, activations, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("img_shape", [(3, 32, 32), (10, 32, 32)])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
@pytest.mark.parametrize(
    "activations",
    [qfloat8_e5m2, qfloat8_e4m3fn, qfloat8_e4m3fnuz],
    ids=["a-float8-e5m2", "a-float8-e4m3", "a-float8-e4m3-uz"],
)
@pytest.mark.skip_device("mps")
def test_quantize_conv2d_float32_activations_float8(
    batch_size, img_shape, out_channels, use_bias, weights, activations, device
):
    _test_quantize_conv2d(batch_size, img_shape, out_channels, use_bias, weights, activations, torch.float32, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("img_shape", [(3, 32, 32), (10, 32, 32)])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
def test_quantize_conv2d_float16_weight_only(batch_size, img_shape, out_channels, use_bias, weights, device):
    _test_quantize_conv2d(batch_size, img_shape, out_channels, use_bias, weights, None, torch.float16, device)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("img_shape", [(3, 32, 32), (10, 32, 32)])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
def test_quantize_conv2d_float32_weight_only(batch_size, img_shape, out_channels, use_bias, weights, device):
    _test_quantize_conv2d(batch_size, img_shape, out_channels, use_bias, weights, None, torch.float32, device)


@pytest.mark.parametrize("img_shape", [(3, 32, 32), (10, 32, 32)])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("activations", [None, qint8], ids=["a-float", "a-int8"])
@pytest.mark.parametrize("weights", [qint4, qint8], ids=["w-int4", "w-int8"])
def test_qconv2d_gradient(img_shape, out_channels, activations, weights, device):
    batch_size = 10
    conv2d = torch.nn.Conv2d(img_shape[0], out_channels, kernel_size=3, bias=True).to(device)
    qconv2d = QConv2d.from_module(conv2d, weights=weights, activations=activations)
    assert qconv2d.weight.requires_grad is True
    assert qconv2d.bias.requires_grad is True
    # Run an inference with identical inputs
    qinputs = random_qactivation((batch_size,) + img_shape, dtype=torch.float32).to(device)
    qout = qconv2d(qinputs)
    out = conv2d(qinputs.dequantize())
    # Outputs are not identical because of the quantization
    assert not torch.equal(qout, out)
    # Compute gradients and compare
    gradient = torch.randn(qout.size()).to(device)
    qout.backward(gradient)
    out.backward(gradient)
    # Gradients are nearly identical because they depend only on the input
    atol = 1e-5
    assert_similar(qconv2d.weight.grad, conv2d.weight.grad, atol=atol)
    assert_similar(qconv2d.bias.grad, conv2d.bias.grad, atol=atol)
