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
