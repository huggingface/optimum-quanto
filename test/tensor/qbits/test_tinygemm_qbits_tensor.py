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
from helpers import assert_similar, device_eq, random_qbits_tensor

from optimum.quanto import TinyGemmQBitsTensor, qint4


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
def test_tinygemm_qbits_tensor_from_qbits_tensor(in_features, out_features, device):
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip(reason="CUDA device >= sm80 not available")
    qtype = qint4
    group_size = 128
    dtype = torch.bfloat16
    shape = (out_features, in_features)
    qbt = random_qbits_tensor(shape, qtype, dtype, group_size, device)
    # Create a TinyGemmQBitsTensor from QBitsTensor
    tgqbt = TinyGemmQBitsTensor(
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
    # Verify that we can reconstruct the QBitsTensor
    new_qbt = tgqbt.qbits_tensor()
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
def test_tinygemm_qbits_tensor_move(device):
    qtype = qint4
    group_size = 128
    dtype = torch.bfloat16
    shape = (1024, 1024)
    # Create a TinyGemmQBitsTensor from a QBitsTensor on CPU
    qbt = random_qbits_tensor(shape, qtype, dtype, group_size, device=torch.device("cpu"))
    tgqbt_cpu = TinyGemmQBitsTensor(
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
    assert tgqbt.dtype == tgqbt_cpu.dtype
    assert tgqbt.qtype == tgqbt_cpu.qtype
    assert tgqbt.shape == tgqbt_cpu.shape
    assert torch.equal(tgqbt.dequantize().cpu(), tgqbt_cpu.dequantize())
