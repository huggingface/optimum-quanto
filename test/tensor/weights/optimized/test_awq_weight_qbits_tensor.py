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
from helpers import assert_similar, device_eq, random_tensor, random_weight_qbits_tensor

from optimum.quanto import qint4
from optimum.quanto.tensor.weights import WeightQBitsTensor
from optimum.quanto.tensor.weights.awq import AWQWeightQBitsTensor


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8, reason="CUDA >= sm80 not available"
)
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
def test_awq_weight_qbits_tensor_from_qbits_tensor(in_features, out_features):
    qtype = qint4
    group_size = 128
    dtype = torch.float16
    shape = (out_features, in_features)
    device = torch.device("cuda")
    qbt = random_weight_qbits_tensor(shape, qtype, dtype, group_size, device)
    # Create a AWQWeightQBitsTensor from the WeightQBitsTensor members
    awqbt = AWQWeightQBitsTensor(
        qtype=qbt.qtype,
        axis=qbt.axis,
        group_size=qbt._group_size,
        size=qbt.size(),
        stride=qbt.stride(),
        data=qbt._data.unpack(),
        scale=qbt._scale,
        shift=qbt._shift,
    )
    assert awqbt.dtype == dtype
    assert awqbt.qtype == qtype
    assert awqbt.shape == shape
    assert device_eq(awqbt.device, device)
    # Verify the dequantized tensors are identical
    assert torch.equal(awqbt.dequantize(), qbt.dequantize())
    # Now verify that we can reconstruct the WeightQBitsTensor
    new_qbt = awqbt.weight_qbits_tensor()
    assert type(new_qbt) is WeightQBitsTensor
    assert new_qbt.dtype == dtype
    assert new_qbt.qtype == qtype
    assert new_qbt.shape == shape
    assert torch.equal(new_qbt._data, qbt._data)
    assert torch.equal(new_qbt._scale, qbt._scale)
    assert torch.equal(new_qbt._shift, qbt._shift)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_awq_weight_qbits_tensor_move(device):
    qtype = qint4
    group_size = 128
    dtype = torch.float16
    shape = (1024, 1024)
    device = torch.device("cuda")
    # Create an AWQWeightQBitsTensor from a QBitsTensor on CUDA
    qbt = random_weight_qbits_tensor(shape, qtype, dtype, group_size, device=torch.device("cuda"))
    awqbt = AWQWeightQBitsTensor(
        qtype=qbt.qtype,
        axis=qbt.axis,
        group_size=qbt._group_size,
        size=qbt.size(),
        stride=qbt.stride(),
        data=qbt._data.unpack(),
        scale=qbt._scale,
        shift=qbt._shift,
    )
    # Move to device, dequantize and compare
    moved_qbt = awqbt.to(device)
    assert isinstance(moved_qbt, WeightQBitsTensor)
    if device.type != "cuda":
        assert type(moved_qbt) is not AWQWeightQBitsTensor
    assert awqbt.dtype == moved_qbt.dtype
    assert awqbt.qtype == moved_qbt.qtype
    assert awqbt.shape == moved_qbt.shape
    assert torch.equal(awqbt.dequantize().to(device), moved_qbt.dequantize())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(256, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_awq_weight_qbits_tensor_linear(batch_size, tokens, embeddings, use_bias):
    device = torch.device("cuda")
    dtype = torch.float16
    weight_qtype = qint4
    group_size = 128
    inputs = torch.rand((batch_size,) + (tokens, embeddings), dtype=dtype, device=device)
    # Create an AWQWeightQBitsTensor from a QBitsTensor on CUDA
    qbt = random_weight_qbits_tensor(
        (embeddings, embeddings), weight_qtype, dtype, group_size, device=torch.device("cuda")
    )
    awq_qweight = AWQWeightQBitsTensor(
        qtype=qbt.qtype,
        axis=qbt.axis,
        group_size=qbt._group_size,
        size=qbt.size(),
        stride=qbt.stride(),
        data=qbt._data.unpack(),
        scale=qbt._scale,
        shift=qbt._shift,
    )
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, awq_qweight, bias)
    out = torch.nn.functional.linear(inputs, qbt.dequantize(), bias)
    assert_similar(out, qout)
