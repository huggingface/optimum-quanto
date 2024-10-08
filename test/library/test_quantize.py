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
    MaxOptimizer,
    absmax_scale,
    qfloat8,
    qfloat8_e4m3fn,
    qfloat8_e4m3fnuz,
    qfloat8_e5m2,
    qint2,
    qint4,
    qint8,
)
from optimum.quanto.tensor.grouped import ungroup


@pytest.mark.parametrize("input_shape", [(32, 32), (32, 10, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8], ids=["qint8"])
@pytest.mark.parametrize(
    "axis",
    [None, 0, -1],
    ids=["per-tensor", "first-axis", "last-axis"],
)
def test_symmetric_quantize_int(input_shape, dtype, qtype, axis, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, qtype=qtype, axis=axis)
    data = torch.ops.quanto.quantize_symmetric(a, dtype=qtype.dtype, axis=axis, scale=scale)
    assert data.dtype == qtype.dtype
    assert device_eq(data.device, device)
    assert_similar(a, data * scale)


@pytest.mark.skip_device("mps")
@pytest.mark.parametrize("input_shape", [(32, 32), (32, 10, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize(
    "qtype",
    [qfloat8, qfloat8_e4m3fn, qfloat8_e4m3fnuz, qfloat8_e5m2],
    ids=["qfloat8", "qfloat8_e4m3fn", "qfloat8_e4m3fnuz", "qfloat8_e5m2"],
)
@pytest.mark.parametrize(
    "axis",
    [None, 0, -1],
    ids=["per-tensor", "first-axis", "last-axis"],
)
def test_symmetric_quantize_float8(input_shape, dtype, qtype, axis, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, qtype=qtype, axis=axis)
    data = torch.ops.quanto.quantize_symmetric(a, dtype=qtype.dtype, axis=axis, scale=scale)
    assert data.dtype == qtype.dtype
    assert device_eq(data.device, device)
    assert_similar(a, data.to(dtype) * scale, atol=5e-3)


@pytest.mark.parametrize("input_shape", [(32, 32), (32, 10, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [None, 8], ids=["channel-wise", "group-wise"])
@pytest.mark.parametrize("shift_mode", ["zeropoint", "float"])
def test_affine_quantize(input_shape, dtype, qtype, axis, group_size, shift_mode, device):
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale, shift = MaxOptimizer()(a, qtype=qtype, axis=axis, group_size=group_size)
    if shift_mode == "zeropoint":
        shift = torch.round(shift / scale).to(torch.int8)
    data = torch.ops.quanto.quantize_affine(a, qtype.bits, axis, group_size, scale, shift)
    assert data.dtype == torch.uint8
    assert device_eq(data.device, device)
    if shift_mode == "zeropoint":
        qa = (data - shift) * scale
    else:
        qa = data * scale - shift
    atol = {
        qint4: {
            "zeropoint": 4e-3,
            "float": 3e-3,
        },
        qint2: {
            "zeropoint": 6e-2,
            "float": 5e-2,
        },
    }[qtype][shift_mode]
    if group_size is not None:
        qa = ungroup(qa, axis=axis, orig_shape=a.shape)
    assert_similar(a, qa, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
def test_affine_quantize_integer_tensor(dtype, qtype, device):
    """This test verifies that an integer tensor in the correct range is preserved."""
    bits = qtype.bits
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    a = torch.tensor(range(qmin, qmax + 1), dtype=dtype).to(device)
    scale, shift = MaxOptimizer()(a, qtype=qtype, axis=0, group_size=None)
    zeropoint = torch.round(shift / scale)
    data = torch.ops.quanto.quantize_affine(a, bits, 0, None, scale, zeropoint)

    assert data.dtype == torch.uint8
    assert device_eq(data.device, device)
    assert torch.equal(a, data - zeropoint)
