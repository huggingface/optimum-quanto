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

from contextlib import nullcontext

import pytest
import torch

from optimum.quanto.library import disable_extensions
from optimum.quanto.tensor.qbits.packed import pack_weights


@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
@pytest.mark.parametrize("shape", [(12,), (32, 32)], ids=["vector", "matrix"])
@pytest.mark.parametrize("use_ext", [True, False], ids=["ext", "no-ext"])
def test_unpack(bits, shape, use_ext, device):
    qmax = 2**bits
    a = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    packed_a = pack_weights(a, bits)
    context = nullcontext() if use_ext else disable_extensions()
    with context:
        unpacked_a = torch.ops.quanto.unpack(packed_a, bits)
    assert unpacked_a.dtype == torch.uint8
    assert torch.equal(unpacked_a, a)
