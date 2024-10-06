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
from pack_intweight import pack_intweight
from packing_utils import pack_awq, reverse_awq_order, unpack_awq

from optimum.quanto import AWQPackedTensor, AWQPacking


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("reorder", [True, False])
@pytest.mark.parametrize("random", [True, False])
def test_awq_pack(in_features, out_features, reorder, random):
    """This test verifies two things:

    - that we are able to replicate awq packing,
    - that we can unpack awq packed tensors and recover the original tensor.
    """
    bits = 4
    interleave = 4
    kstride = 64
    qmax = 2**bits
    shape = (out_features, in_features)
    device = torch.device('cuda')
    if random:
        t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    else:
        numel = np.prod(shape)
        t = torch.tensor(range(numel), dtype=torch.int32)
        t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    packed = pack_awq(t.to(torch.int32), reorder=reorder)
    # Sanity check: verify we can recover the Tensor using AWQ unpacking
    unpacked = unpack_awq(packed, bits=4)
    if reorder:
        unpacked = reverse_awq_order(unpacked, bits=4)
    unpacked = torch.bitwise_and(unpacked, qmax - 1)
    assert torch.equal(t, unpacked)
    # Compare with quanto packing
    repacked = AWQPackedTensor.pack(t, packing=AWQPacking.V1, reorder=reorder)
    assert torch.equal(packed, repacked._data)
    unpacked = repacked.unpack()
    assert torch.equal(unpacked, t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("random", [True, False])
def test_awq_pack_v2(in_features, out_features, random):
    """This test verifies two things:

    - that we are able to replicate awq packing,
    - that we can unpack awq packed tensors and recover the original tensor.
    """
    bits = 4
    interleave = 4
    kstride = 64
    qmax = 2**bits
    shape = (out_features, in_features)
    device = torch.device('cuda')
    if random:
        t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    else:
        numel = np.prod(shape)
        t = torch.tensor(range(numel), dtype=torch.int32)
        t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    packed = pack_intweight(t.to(torch.int32), interleave=interleave, kstride=kstride)
    # Compare with quanto packing
    repacked = AWQPackedTensor.pack(t, packing=AWQPacking.V2)
    assert torch.equal(packed, repacked._data)
    unpacked = repacked.unpack()
    assert torch.equal(unpacked, t)

