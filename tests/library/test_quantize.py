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
from helpers import assert_similar, device_eq, random_tensor, torch_min_version

from optimum.quanto import (
    AbsmaxOptimizer,
    MaxOptimizer,
    absmax_scale,
    qfloat8,
    qfloat8_e4m3fn,
    qfloat8_e4m3fnuz,
    qfloat8_e5m2,
    qint2,
    qint4,
    qint8,
    quantize_weight,
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


@torch_min_version("2.6.0")
def test_symmetric_quantize_dtensor_sharding():
    dist = pytest.importorskip("torch.distributed")
    dtensor = pytest.importorskip("torch.distributed.tensor")
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    import os
    import tempfile

    init_file = tempfile.NamedTemporaryFile(delete=False)
    init_file.close()
    try:
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{init_file.name}")
        mesh = dtensor.DeviceMesh("cpu", [0])
        base = random_tensor((4, 4), dtype=torch.float32)
        scale = absmax_scale(base, qtype=qint8, axis=0)
        local_expected = torch.ops.quanto.quantize_symmetric(base, dtype=qint8.dtype, axis=0, scale=scale)

        sharded_base = dtensor.distribute_tensor(base, mesh, [dtensor.Shard(0)])
        sharded_scale = dtensor.distribute_tensor(scale, mesh, [dtensor.Shard(0)])

        data = torch.ops.quanto.quantize_symmetric(sharded_base, dtype=qint8.dtype, axis=0, scale=sharded_scale)

        assert data.placements == (dtensor.Shard(0),)
        assert data.to_local().dtype == qint8.dtype
        assert torch.equal(data.to_local(), local_expected)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(init_file.name):
            os.unlink(init_file.name)


@torch_min_version("2.6.0")
def test_qbytes_linear_dtensor_weight_sharding():
    dist = pytest.importorskip("torch.distributed")
    dtensor = pytest.importorskip("torch.distributed.tensor")
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    import os
    import tempfile

    init_file = tempfile.NamedTemporaryFile(delete=False)
    init_file.close()
    try:
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{init_file.name}")
        mesh = dtensor.DeviceMesh("cpu", [0])
        weight = random_tensor((4, 4), dtype=torch.float32)
        scale = AbsmaxOptimizer()(weight, qtype=qint8, axis=0)
        input = random_tensor((2, 4), dtype=torch.float32)
        bias = random_tensor((4,), dtype=torch.float32)
        local_qweight = quantize_weight(weight, qtype=qint8, axis=0, scale=scale, optimized=False)
        local_input = input.clone().requires_grad_()
        local_expected = torch.nn.functional.linear(local_input, local_qweight, bias=bias)

        sharded_qweight = quantize_weight(
            dtensor.distribute_tensor(weight, mesh, [dtensor.Shard(0)]),
            qtype=qint8,
            axis=0,
            scale=dtensor.distribute_tensor(scale, mesh, [dtensor.Shard(0)]),
            optimized=False,
        )

        sharded_input = input.clone().requires_grad_()
        output = torch.nn.functional.linear(sharded_input, sharded_qweight, bias=bias)

        assert output.placements == (dtensor.Shard(1),)
        assert torch.equal(output.to_local(), local_expected)
        local_expected.sum().backward()
        output.full_tensor().sum().backward()
        assert torch.equal(sharded_input.grad, local_input.grad)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(init_file.name):
            os.unlink(init_file.name)


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
