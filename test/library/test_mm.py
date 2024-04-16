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
from helpers import assert_similar, random_tensor

from quanto import AWQPackedTensor, AWQPacking


@pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
@pytest.mark.parametrize("output_features", [48, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dqmm(input_shape, output_features, dtype, device):
    input = random_tensor(input_shape, dtype=dtype).to(device)
    other = torch.randint(-127, 127, (input_shape[-1], output_features), dtype=torch.int8).to(device)
    other_scale = random_tensor((output_features,), dtype=dtype).to(device)
    output = torch.ops.quanto.dqmm(input, other, other_scale)
    expected = torch.ops.aten.mm(input, other * other_scale)
    assert torch.equal(expected, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features, out_features", [(256, 256), (512, 256)])
@pytest.mark.parametrize("batch_size, tokens", [(4, 1), (10, 128)], ids=["gemv", "gemm"])
def test_gemm_fp16_int4(batch_size, tokens, in_features, out_features):
    """This test verifies that the GEMM operation is equivalent to torch.mm."""
    bits = 4
    group_size = 128  # Hard-coded in kernels
    device = torch.device("cuda")
    input_shape = (batch_size, tokens, in_features)
    # FIXME: does not work if inputs are negative !!??
    inputs = torch.rand(input_shape, dtype=torch.float16, device=device)
    qmax = 2**bits
    other_shape = (out_features, in_features)
    other_data = torch.randint(0, qmax, other_shape, dtype=torch.uint8, device=device)
    packed_other_data = AWQPackedTensor.pack(other_data, packing=AWQPacking.V2)._data
    # The GEMM kernel works on transposed scales
    scales_shape = (in_features // group_size, out_features)
    other_scales = torch.rand(scales_shape, dtype=torch.float16, device=device) / qmax
    # The GEMM kernel works on transposed, negated and scaled zeropoints
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1)
    other_zeropoints = torch.randint(qmin, qmax, scales_shape, dtype=torch.int8, device=device)
    # Negate and scale
    other_scaled_zeropoints = -other_zeropoints * other_scales
    # Evaluate mm outputs using the GEMM kernel
    lib_outputs = torch.ops.quanto.gemm(
        inputs,
        packed_other_data,
        other_scales,
        other_scaled_zeropoints,
        rows=inputs.numel() // inputs.shape[-1],
        out_cols=out_features,
        in_cols=in_features,
        bits=4,
        group_size=group_size,
    )
    # Transpose other data and reshape it to align it with transposed scales and zeros
    other_data_t = other_data.t().reshape(group_size, in_features // group_size, out_features)
    # Dequantize transposed other
    other_t = (other_data_t - other_zeropoints) * other_scales
    # Reshape it as expected by the matmul
    other_t = other_t.reshape(in_features, out_features)
    # Evaluate the matrix multiplication using pytorch float16 mm
    pt_outputs = torch.matmul(inputs, other_t)
    # Verify the results are similar
    assert_similar(lib_outputs, pt_outputs, rtol=5e-3)
