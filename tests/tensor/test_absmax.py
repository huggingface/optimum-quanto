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
from helpers import random_tensor

from optimum.quanto import absmax_scale, qfloat8, qint8


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (2, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_absmax_scale(input_shape, axis, dtype, qtype, device):
    if device.type == "mps" and qtype.is_floating_point:
        pytest.skip("Float8 are not supported on MPS device")
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, qtype, axis)
    assert scale.dtype == dtype
    if axis is None:
        assert scale.ndim == 0
    else:
        assert scale.ndim == a.ndim
        sscale = torch.squeeze(scale)
        if a.ndim == 1 or a.shape[axis] == 1:
            # Quantization is actually per-tensor as the axis dim is 1
            assert sscale.ndim == 0
        else:
            assert sscale.ndim == 1
