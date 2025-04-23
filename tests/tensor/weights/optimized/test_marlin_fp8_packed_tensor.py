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

from optimum.quanto.library.extensions import is_extension_available
from optimum.quanto.tensor.weights.marlin.fp8 import MarlinF8PackedTensor


def get_fp8_tensor(shape, device, random=False):
    # We will initialize float8 from an uint8 tensor
    qmax = 2**8
    if random:
        t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    else:
        numel = np.prod(shape)
        t = torch.tensor(range(numel), dtype=torch.int32)
        t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    # Remove values that would be interpreted as nans in float8.
    t[t == 127] = 0
    t[t == 255] = 0
    return t.view(torch.float8_e4m3fn).to(device)


@pytest.mark.skipif(not is_extension_available("quanto_cuda"), reason="CUDA extension is not available")
@pytest.mark.parametrize("in_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("random", [True, False])
def test_pack_marlin_fp8_tensor(in_features, out_features, random):
    shape = (out_features, in_features)
    device = torch.device("cuda")
    t = get_fp8_tensor(shape, device, random)
    packed = MarlinF8PackedTensor.pack(t)
    assert isinstance(packed, MarlinF8PackedTensor)
    assert device_eq(packed.device, device)
    assert torch.equal(t, packed.unpack())


@pytest.mark.skipif(not is_extension_available("quanto_cuda"), reason="CUDA extension is not available")
def test_move_marlin_fp8_tensor():
    shape = (256, 256)
    device = torch.device("cuda")
    t = get_fp8_tensor(shape, device)
    packed = MarlinF8PackedTensor.pack(t)
    moved = packed.to("cuda")
    assert isinstance(moved, MarlinF8PackedTensor)
    # Marlin FP8 tensors are unpacked when moved out of CUDA device
    moved = packed.to("cpu")
    assert type(moved) is torch.Tensor
    assert torch.equal(t, moved.to("cuda"))
