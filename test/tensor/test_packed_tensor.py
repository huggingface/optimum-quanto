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

import io

import pytest
import torch
from helpers import device_eq

from quanto import PackedTensor


@pytest.mark.parametrize("shape", [(10,), (12,), (10, 10), (12, 10), (32, 32)])
@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
def test_pack_tensor(shape, bits, device):
    """This test verifies that an integer tensor in the correct range is preserved."""
    qmax = 2**bits
    t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    packed = PackedTensor.pack(t, bits=bits)

    assert isinstance(packed, PackedTensor)
    assert packed.dtype == torch.uint8
    assert device_eq(packed.device, device)
    assert torch.equal(t, packed.unpack())


@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
def test_packed_tensor_serialization(bits, device):
    qmax = 2**bits
    shape = (10, 32)
    t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    packed = PackedTensor.pack(t, bits=bits)
    b = io.BytesIO()
    torch.save(packed, b)
    b.seek(0)
    packed_reloaded = torch.load(b)
    assert isinstance(packed_reloaded, PackedTensor)
    assert packed_reloaded.shape == packed.shape
    assert packed_reloaded.dtype == packed.dtype
    assert packed_reloaded.bits == packed.bits
    assert torch.equal(packed_reloaded._data, packed._data)
    assert torch.equal(t, packed_reloaded.unpack())
