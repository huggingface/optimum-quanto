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
from helpers import random_tensor, torch_min_version

from optimum.quanto import ActivationQBytesTensor, absmax_scale, qint8, quantize_activation


def compile_for_device(f, device):
    # Remove any side-effects form previous compilation
    torch.compiler.reset()
    # Inductor relies on Triton for inference which does not support MPS
    backend = "aot_eager" if device == torch.device("mps") else "aot_eager"
    return torch.compile(f, backend=backend)


@torch_min_version("2.7.0")
@pytest.mark.parametrize("input_shape", [(2, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8], ids=["qint8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
def test_compile_quantize_tensor(input_shape, qtype, dtype, device):
    if device == torch.device("mps") and dtype == torch.bfloat16:
        pytest.skip("BFloat16 is not supported on MPS")
    a = random_tensor(input_shape, dtype=dtype).to(device)

    def f(x, qtype):
        scale = absmax_scale(x)
        return quantize_activation(x, qtype=qtype, scale=scale)

    compiled_f = compile_for_device(f, device)
    qa = compiled_f(a, qtype)
    assert isinstance(qa, ActivationQBytesTensor)
    assert qa.qtype == qtype
    assert qa._scale.dtype == dtype
    assert qa.axis is None


def test_compile_qtensor_to(device):
    input_shape = (10, 32, 32)
    a = random_tensor(input_shape).to(device)

    def f(x, dtype):
        return x.to(dtype)

    compiled_f = compile_for_device(f, device)

    scale = absmax_scale(a)
    qa = quantize_activation(a, qtype=qint8, scale=scale)
    cqa = compiled_f(qa, torch.float16)
    assert isinstance(cqa, ActivationQBytesTensor)
    assert cqa.qtype == qint8
    assert cqa._scale.dtype == torch.float16
