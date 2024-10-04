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
from packaging import version

from optimum.quanto.tensor.weights.tinygemm import TinyGemmPackedTensor


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("random", [True, False])
def test_pack_tinygemm_tensor(in_features, out_features, random, device):
    if device.type == "cuda":
        if torch.version.hip:
            pytest.skip(reason="TinyGemm is not supported on ROCm devices")
        if version.parse(torch.version.cuda).release < (12, 1):
            pytest.skip(reason="CUDA runtime must be at least 12.1")
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip(reason="CUDA device >= sm80 not available")
    bits = 4
    qmax = 2**bits
    shape = (out_features, in_features)
    if random:
        t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    else:
        numel = np.prod(shape)
        t = torch.tensor(range(numel), dtype=torch.int32)
        t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    packed = TinyGemmPackedTensor.pack(t)
    assert isinstance(packed, TinyGemmPackedTensor)
    assert device_eq(packed.device, device)
    assert torch.equal(t, packed.unpack())


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
def test_move_tinygemm_packed_tensor(device):
    if device.type == "cuda":
        if torch.version.hip:
            pytest.skip(reason="TinyGemm is not supported on ROCm devices")
        if version.parse(torch.version.cuda).release < (12, 1):
            pytest.skip(reason="CUDA runtime must be at least 12.1")
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip(reason="CUDA device >= sm80 not available")
    shape = (256, 256)
    bits = 4
    qmax = 2**bits
    numel = np.prod(shape)
    t = torch.tensor(range(numel), dtype=torch.int32)
    t = (t % qmax).reshape(shape).to(torch.uint8)
    packed = TinyGemmPackedTensor.pack(t)
    moved = packed.to(device)
    assert torch.equal(t.to(device), moved.unpack())
