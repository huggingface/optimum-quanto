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
from helpers import assert_similar, random_qactivation, random_qweight, random_tensor

from optimum.quanto import qint2, qint4, qint8, qtypes


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("activation_qtype", [None, qint8], ids=["a-none", "a-qint8"])
@pytest.mark.parametrize("weight_qtype", [qint2, qint4, qint8], ids=["w-qint2", "w-qint4", "w-qint8"])
def test_qactivation_qweight_linear(
    batch_size, tokens, embeddings, use_bias, dtype, activation_qtype, weight_qtype, device
):
    input_shape = (batch_size, tokens, embeddings)
    if activation_qtype is None:
        inputs = random_tensor(input_shape, dtype=dtype).to(device)
    else:
        inputs = random_qactivation(input_shape, qtype=activation_qtype, dtype=dtype).to(device)
    qweight = random_qweight((embeddings, embeddings), qtype=weight_qtype, dtype=dtype, axis=0).to(device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    if activation_qtype is not None:
        inputs = inputs.dequantize()
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    assert_similar(out, qout)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(256, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_linear_fp16_int4(batch_size, tokens, embeddings, use_bias, device):
    dtype = torch.float16
    weight_qtype = qint4
    inputs = torch.rand((batch_size,) + (tokens, embeddings), dtype=dtype, device=device)
    qweight = random_qweight((embeddings, embeddings), weight_qtype, dtype=dtype, axis=0, group_size=128).to(device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    assert_similar(out, qout)


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(256, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_linear_bf16_int4(batch_size, tokens, embeddings, use_bias, device):
    dtype = torch.bfloat16
    weight_qtype = qint4
    input_shape = (batch_size, tokens, embeddings)
    inputs = torch.rand(input_shape, dtype=dtype, device=device)
    weight_shape = (embeddings, embeddings)
    qweight = random_qweight(weight_shape, weight_qtype, dtype=dtype, axis=0, group_size=128, device=device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    assert_similar(out, qout)


@pytest.mark.parametrize("activation_qtype", [qint8, qtypes["qfloat8_e4m3fn"]])
def test_linear_fp16_scale_underflow(activation_qtype, device):
    if device.type == "mps" and activation_qtype.is_floating_point:
        pytest.skip("Float8 is not supported on MPS")
    g = torch.Generator(device=device).manual_seed(0)
    rows = torch.tensor([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0], device=device)
    w = torch.randn(8, 1024, generator=g, device=device)
    w = w / w.abs().amax(1, keepdim=True) * rows.view(8, 1)
    x = (torch.randn(32, 1024, generator=g, device=device) / 30.0).half()

    from optimum.quanto import Calibration, freeze, quantize

    m = torch.nn.Sequential(torch.nn.Linear(1024, 8, bias=False)).half().to(device)
    m[0].weight.data.copy_(w)
    quantize(m, weights=qint8, activations=activation_qtype)
    with Calibration():
        m(x)
    freeze(m)
    with torch.no_grad():
        y = m(x)

    # Ensure no output column underflows to all zeros
    assert not (y == 0).all(dim=0).any(), f"Zero columns detected for activation {activation_qtype.name}"
    assert y.dtype == torch.float16
