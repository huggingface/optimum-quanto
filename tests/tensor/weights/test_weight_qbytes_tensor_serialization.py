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
from helpers import random_qweight

from optimum.quanto import qfloat8, qint8


@pytest.mark.parametrize("input_shape", [(10, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_weights_qbytes_tensor_serialization(input_shape, qtype, dtype, axis):
    qinputs = random_qweight(input_shape, dtype=dtype, qtype=qtype, axis=axis)
    b = io.BytesIO()
    torch.save(qinputs, b)
    b.seek(0)
    qinputs_reloaded = torch.load(b, weights_only=False)
    assert qinputs_reloaded.qtype == qtype
    assert torch.equal(qinputs_reloaded._scale, qinputs._scale)
    if qtype.is_floating_point:
        # Equality is not supported for float8
        assert torch.equal(qinputs_reloaded._data.to(torch.float32), qinputs._data.to(torch.float32))
    else:
        assert torch.equal(qinputs_reloaded._data, qinputs._data)
    # We cannot test dtype directly as it is not correctly set by torch.load
    assert qinputs_reloaded._scale.dtype == dtype
    assert qinputs_reloaded.axis == qinputs.axis
