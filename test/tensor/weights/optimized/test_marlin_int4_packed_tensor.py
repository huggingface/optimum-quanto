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


import numpy as np
import pytest
import torch
from helpers import device_eq

from optimum.quanto.tensor.weights.marlin.int4 import MarlinInt4PackedTensor


def get_uint4_tensor(shape, device, random=False):
    qmax = 2**4
    if random:
        t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    else:
        numel = np.prod(shape)
        t = torch.tensor(range(numel), dtype=torch.int32)
        t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    return t


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("random", [True, False])
def test_pack_marlin_int4_tensor(in_features, out_features, random):
    shape = (out_features, in_features)
    device = torch.device("cuda")
    t = get_uint4_tensor(shape, device, random)
    packed = MarlinInt4PackedTensor.pack(t)
    assert isinstance(packed, MarlinInt4PackedTensor)
    assert device_eq(packed.device, device)
    assert torch.equal(t, packed.unpack())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_marlin_int4_packed_tensor(device):
    shape = (256, 256)
    device = torch.device("cuda")
    t = get_uint4_tensor(shape, device)
    packed = MarlinInt4PackedTensor.pack(t)
    moved = packed.to("cuda")
    assert isinstance(moved, MarlinInt4PackedTensor)
    # Marlin int4 tensors are unpacked when moved out of CUDA device
    moved = packed.to("cpu")
    assert type(moved) is torch.Tensor
    assert torch.equal(t, moved.to("cuda"))
