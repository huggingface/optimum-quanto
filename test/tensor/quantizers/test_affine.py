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
from helpers import assert_similar, device_eq, random_tensor

from optimum.quanto import (
    AffineQuantizer,
    MaxOptimizer,
    QBitsTensor,
    qint2,
    qint4,
)


@pytest.mark.parametrize("input_shape", [(32, 32), (32, 10, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [None, 8], ids=["channel-wise", "group-wise"])
def test_affine_quantize(input_shape, dtype, qtype, axis, group_size, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale, zeropoint = MaxOptimizer()(a, bits=qtype.bits, axis=axis, group_size=group_size)
    qa = AffineQuantizer.apply(a, qtype, axis, group_size, scale, zeropoint)
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    atol = 5e-3 if qtype == qint4 else 6e-2
    assert_similar(a, qa, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
def test_affine_quantize_integer_tensor(dtype, qtype, device):
    """This test verifies that an integer tensor in the correct range is preserved."""
    bits = qtype.bits
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    a = torch.tensor(range(qmin, qmax + 1), dtype=dtype).to(device)
    scale, zeropoint = MaxOptimizer()(a, bits=qtype.bits, axis=0, group_size=None)
    qa = AffineQuantizer.apply(a, qtype, 0, None, scale, zeropoint)

    assert qa._data.dtype == torch.uint8
    assert isinstance(qa, QBitsTensor)
    assert qa.dtype == dtype
    assert qa.qtype == qtype
    assert device_eq(qa.device, device)
    assert torch.equal(a, qa.dequantize())
