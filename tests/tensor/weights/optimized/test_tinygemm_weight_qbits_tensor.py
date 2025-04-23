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
from helpers import assert_similar, device_eq, random_qweight, random_tensor
from packaging import version

from optimum.quanto import qint4
from optimum.quanto.tensor.weights import WeightQBitsTensor
from optimum.quanto.tensor.weights.tinygemm import TinyGemmWeightQBitsTensor


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
def test_tinygemm_weight_qbits_tensor_from_qbits_tensor(in_features, out_features, device):
    if device.type == "cuda":
        if torch.version.hip:
            pytest.skip(reason="TinyGemm not available for ROCm devices")
        if version.parse(torch.version.cuda).release < (12, 1):
            pytest.skip(reason="CUDA runtime must be at least 12.1")
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip(reason="CUDA device >= sm80 not available")
    qtype = qint4
    group_size = 128
    dtype = torch.bfloat16
    shape = (out_features, in_features)
    qbt = random_qweight(shape, qtype, dtype, group_size=group_size, device=device)
    # Create a TinyGemmWeightQBitsTensor from the WeightQBitsTensor members
    tgqbt = TinyGemmWeightQBitsTensor(
        qtype=qbt.qtype,
        axis=qbt.axis,
        group_size=qbt._group_size,
        size=qbt.size(),
        stride=qbt.stride(),
        data=qbt._data.unpack(),
        scale_shift=(qbt._scale, qbt._shift),
    )
    assert tgqbt.dtype == dtype
    assert tgqbt.qtype == qtype
    assert tgqbt.shape == shape
    assert device_eq(tgqbt.device, device)
    # Verify that we can reconstruct the WeightQBitsTensor
    new_qbt = tgqbt.weight_qbits_tensor()
    assert type(new_qbt) is WeightQBitsTensor
    assert new_qbt.dtype == dtype
    assert new_qbt.qtype == qtype
    assert new_qbt.shape == shape
    assert torch.equal(new_qbt._data, qbt._data)
    assert torch.equal(new_qbt._scale, qbt._scale)
    # FIXME: we cannot guarantee an exact match because of the addition/removal of the mid-point
    # which is lossy in bfloat16 (a + b - b != a)
    assert_similar(new_qbt._shift, qbt._shift)
    # Verify the dequantized tensors are similar
    assert_similar(tgqbt.dequantize(), qbt.dequantize())


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
def test_tinygemm_weight_qbits_tensor_move(device):
    qtype = qint4
    group_size = 128
    dtype = torch.bfloat16
    shape = (1024, 1024)
    # Create a TinyGemmWeightQBitsTensor from a QBitsTensor on CPU
    qbt = random_qweight(shape, qtype, dtype, group_size=group_size, device=torch.device("cpu"))
    tgqbt_cpu = TinyGemmWeightQBitsTensor(
        qtype=qbt.qtype,
        axis=qbt.axis,
        group_size=qbt._group_size,
        size=qbt.size(),
        stride=qbt.stride(),
        data=qbt._data.unpack(),
        scale_shift=(qbt._scale, qbt._shift),
    )
    # Move to device, dequantize and compare
    tgqbt = tgqbt_cpu.to(device)
    assert isinstance(tgqbt, WeightQBitsTensor)
    assert tgqbt.dtype == tgqbt_cpu.dtype
    assert tgqbt.qtype == tgqbt_cpu.qtype
    assert tgqbt.shape == tgqbt_cpu.shape
    assert torch.equal(tgqbt.dequantize().cpu(), tgqbt_cpu.dequantize())


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens", [256, 512])
@pytest.mark.parametrize("embeddings", [256, 512, 1024, 4096])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_tinygemm_weight_qbits_tensor_linear(batch_size, tokens, embeddings, use_bias, device):
    if device.type == "cuda":
        if torch.version.hip:
            pytest.skip(reason="TinyGemm not available for ROCm devices")
        if version.parse(torch.version.cuda).release < (12, 1):
            pytest.skip(reason="CUDA runtime must be at least 12.1")
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip(reason="CUDA device >= sm80 not available")
    qtype = qint4
    group_size = 128
    dtype = torch.bfloat16
    inputs = torch.rand((batch_size,) + (tokens, embeddings), dtype=dtype, device=device)
    # Create a TinyGemmWeightQBitsTensor from a QBitsTensor
    qbt = random_qweight((tokens, embeddings), qtype, dtype, group_size=group_size, device=device)
    tinygemm_qweight = TinyGemmWeightQBitsTensor(
        qtype=qbt.qtype,
        axis=qbt.axis,
        group_size=qbt._group_size,
        size=qbt.size(),
        stride=qbt.stride(),
        data=qbt._data.unpack(),
        scale_shift=(qbt._scale, qbt._shift),
    )
    bias = random_tensor((tokens,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, tinygemm_qweight, bias)
    out = torch.nn.functional.linear(inputs, qbt.dequantize(), bias)
    assert_similar(out, qout)
