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

from optimum.quanto import qfloat8_e4m3fn
from optimum.quanto.library.extensions import is_extension_available
from optimum.quanto.nn import QLinear
from optimum.quanto.tensor.weights.marlin import MarlinF8QBytesTensor


@pytest.mark.skipif(
    not is_extension_available("quanto_cuda") or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA >= sm80 not available",
)
@pytest.mark.parametrize("in_features", [16, 32, 48, 64])
@pytest.mark.parametrize("out_features", [64, 128, 192, 256])
def test_pack_unpack(in_features: int, out_features: int):
    data = torch.randint(0, 256, size=(out_features, in_features), dtype=torch.uint8, device="cuda")

    # Remove nans.
    data[data == 127] = 0
    data[data == 255] = 0

    data = data.view(torch.float8_e4m3fn)

    qtype = qfloat8_e4m3fn
    axis = 0
    size = data.shape
    stride = data.stride()
    scale = torch.rand((out_features, 1), dtype=torch.float16, device="cuda")
    marlin_tensor = MarlinF8QBytesTensor(qtype, axis, size, stride, data, scale)

    data_dequantized = marlin_tensor.dequantize()

    assert torch.all((data.to(torch.float16) * scale - data_dequantized).abs() < 1e-4)


@pytest.mark.skipif(
    not is_extension_available("quanto_cuda") or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA >= sm80 not available",
)
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_marlin_fp8_linear_bias_and_strided_input(batch_size, dtype):
    in_features, out_features = 256, 128
    linear = torch.nn.Linear(in_features, out_features, bias=True).to("cuda", dtype)
    torch.nn.init.normal_(linear.bias)
    qlinear = QLinear.from_module(linear, weights=qfloat8_e4m3fn)
    qlinear.freeze()
    assert isinstance(qlinear.weight, MarlinF8QBytesTensor)

    x = torch.randn(batch_size, in_features, dtype=dtype, device="cuda")
    with torch.no_grad():
        out = qlinear(x)
        ref = torch.nn.functional.linear(x, qlinear.weight.dequantize(), qlinear.bias)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    # A non-contiguous view with the same values must give the same result.
    x_strided = torch.stack([x, x], dim=-1)[..., 0]
    assert not x_strided.is_contiguous()
    with torch.no_grad():
        torch.testing.assert_close(qlinear(x_strided), out)
