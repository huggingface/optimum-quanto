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
from helpers import assert_similar, random_qweight, random_tensor
from tensor.weights.weight_helpers import check_weight_qtensor_linear

from optimum.quanto import MaxOptimizer, QBitsTensor, qint2, qint4, quantize_weight


@pytest.mark.parametrize("group_size", [None, 128], ids=["channel-wise", "group-wise"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_qbitstensor_to_device(dtype, group_size, device):
    qa = random_qweight((256, 512), dtype=dtype, qtype=qint4, group_size=group_size, device="cpu")
    # Keep a copy of the dequantized Tensor as a reference
    dqa = qa.dequantize()
    # Move to the target device
    moved_qa = qa.to(device)
    assert isinstance(moved_qa, QBitsTensor)
    assert moved_qa.device.type == device.type
    assert moved_qa._data.device.type == device.type
    assert moved_qa._scale.device.type == device.type
    assert moved_qa._shift.device.type == device.type
    moved_dqa = moved_qa.dequantize().to("cpu")
    if type(moved_qa) is not QBitsTensor:
        # Since we use an optimized packing, the order of operations during
        # dequantization might differ, but the moved dequantized Tensor should be nearly identical
        assert_similar(moved_dqa, dqa)
    else:
        assert torch.equal(moved_dqa, dqa)


def test_qbitstensor_detach():
    qa = random_qweight((32, 32), qtype=qint4)
    dqa = qa.detach()
    assert isinstance(dqa, QBitsTensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint2, qint4])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_qbitstensor_equal(dtype, qtype, axis, device):
    a = random_tensor((1024, 1024), dtype=dtype, device=device)
    scale, shift = MaxOptimizer()(a, qtype=qtype, axis=axis, group_size=128)
    qa1 = quantize_weight(a, qtype=qtype, axis=axis, scale=scale, shift=shift, group_size=128)
    qa2 = quantize_weight(a, qtype=qtype, axis=axis, scale=scale, shift=shift, group_size=128)
    assert torch.equal(qa1, qa2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens", [16, 32])
@pytest.mark.parametrize("in_features", [256, 512])
@pytest.mark.parametrize("out_features", [256, 512])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_weight_qbits_tensor_linear(dtype, batch_size, tokens, in_features, out_features, use_bias, device):
    weight_qtype = qint4
    group_size = 128
    # Create a QBitsTensor
    qbt = random_qweight((out_features, in_features), weight_qtype, dtype, group_size=group_size, device=device)
    check_weight_qtensor_linear(qbt, batch_size, tokens, use_bias)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens", [16, 32, 48, 64])
@pytest.mark.parametrize("in_features", [1024, 4096, 16384])
@pytest.mark.parametrize("out_features", [1024, 2048, 4096])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_weight_qbits_tensor_linear_gpu(dtype, batch_size, tokens, in_features, out_features, use_bias):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        pytest.skip(reason="Test is too slow on non-GPU devices")

    weight_qtype = qint4
    group_size = 128
    # Create a QBitsTensor
    qbt = random_qweight((out_features, in_features), weight_qtype, dtype, group_size=group_size, device=device)
    check_weight_qtensor_linear(qbt, batch_size, tokens, use_bias)
