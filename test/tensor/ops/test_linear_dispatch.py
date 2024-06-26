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
from helpers import assert_similar, random_qactivation, random_qweight, random_tensor

from optimum.quanto import qint2, qint4, qint8


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("weight_qtype", [qint2, qint4, qint8], ids=["qint2", "qint4", "qint8"])
def test_qactivation_qweight_linear(batch_size, tokens, embeddings, use_bias, dtype, weight_qtype, device):
    qinputs = random_qactivation((batch_size,) + (tokens, embeddings), dtype=dtype).to(device)
    qweight = random_qweight((embeddings, embeddings), weight_qtype, dtype=dtype, axis=0).to(device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    out = torch.nn.functional.linear(qinputs.dequantize(), qweight.dequantize(), bias)
    qout = torch.nn.functional.linear(qinputs, qweight, bias)
    assert_similar(out, qout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(256, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_linear_gemm_fp16_int4(batch_size, tokens, embeddings, use_bias):
    device = torch.device("cuda")
    dtype = torch.float16
    weight_qtype = qint4
    inputs = torch.rand((batch_size,) + (tokens, embeddings), dtype=dtype, device=device)
    qweight = random_qweight((embeddings, embeddings), weight_qtype, dtype=dtype, axis=0, group_size=128).to(device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    assert_similar(out, qout)
