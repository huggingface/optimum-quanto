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

from optimum.quanto import QTensor, qint8, qtypes
from optimum.quanto.nn import QLinear


@pytest.mark.parametrize("in_features", [8, 16])
@pytest.mark.parametrize("out_features", [32, 64])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_qmodule_freeze(in_features, out_features, use_bias, dtype):
    qlinear = QLinear(in_features, out_features, bias=use_bias, weights=qint8).to(dtype)
    assert not qlinear.frozen
    assert not isinstance(qlinear.weight, QTensor)
    assert qlinear.weight.dtype == dtype
    if use_bias:
        assert not isinstance(qlinear.bias, QTensor)
        assert qlinear.bias.dtype == dtype
    qweight = qlinear.qweight
    assert isinstance(qweight, QTensor)
    assert qweight.dtype == dtype
    assert qweight.qtype == qint8
    qlinear.freeze()
    assert qlinear.frozen
    assert isinstance(qlinear.weight, QTensor)
    assert qlinear.weight.dtype == dtype
    assert qlinear.weight.qtype == qint8
    if use_bias:
        assert not isinstance(qlinear.bias, QTensor)
        assert qlinear.bias.dtype == dtype


@pytest.mark.parametrize("weights", ["qint2", "qint4", "qint8", "qfloat8"])
@pytest.mark.parametrize("activations", [None, "qint8", "qfloat8"])
def test_qmodule_qtype_as_string(weights, activations):
    qlinear = QLinear(16, 64, weights=weights, activations=activations)
    assert qlinear.weight_qtype == qtypes[weights]
    assert qlinear.activation_qtype is None if activations is None else qtypes[activations]
