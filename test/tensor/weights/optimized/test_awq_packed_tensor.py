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

from optimum.quanto.tensor.weights.awq import AWQPackedTensor, AWQPacking


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("random", [True, False])
@pytest.mark.parametrize("packing, reorder", [(AWQPacking.V1, True), (AWQPacking.V1, False), (AWQPacking.V2, False)])
def test_pack_awq_tensor(in_features, out_features, random, packing, reorder):
    bits = 4
    qmax = 2**bits
    shape = (out_features, in_features)
    device = torch.device("cuda")
    if random:
        t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    else:
        numel = np.prod(shape)
        t = torch.tensor(range(numel), dtype=torch.int32)
        t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    packed = AWQPackedTensor.pack(t, packing=packing, reorder=reorder)
    assert isinstance(packed, AWQPackedTensor)
    assert packed._packing == packing
    assert packed._reorder == reorder
    assert device_eq(packed.device, device)
    assert torch.equal(t, packed.unpack())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("packing, reorder", [(AWQPacking.V1, True), (AWQPacking.V2, False)])
def test_move_awq_tensor(packing, reorder):
    shape = (256, 256)
    bits = 4
    qmax = 2**bits
    numel = np.prod(shape)
    device = torch.device("cuda")
    t = torch.tensor(range(numel), dtype=torch.int32)
    t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    packed = AWQPackedTensor.pack(t, packing=packing, reorder=reorder)
    assert packed._packing == packing
    assert packed._reorder == reorder
    moved = packed.to("cuda")
    assert isinstance(moved, AWQPackedTensor)
    assert moved._packing == packing
    assert moved._reorder == reorder
    # TensorRT tensors are unpacked when moved out of CUDA device
    moved = packed.to("cpu")
    assert type(moved) is torch.Tensor
