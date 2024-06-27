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

from optimum.quanto import AWQPackedTensor, AWQPacking


@pytest.mark.parametrize("batch_size", [1, 10, None], ids=["single", "batched", "static"])
@pytest.mark.parametrize("input_features", [32, 50])
@pytest.mark.parametrize("output_features", [48, 50, 64])
@pytest.mark.parametrize("input_dtype", [None, torch.int8], ids=["i-as-out", "i-int8"])
@pytest.mark.parametrize("weight_dtype", [torch.float8_e4m3fn, torch.int8], ids=["w-float8", "w-int8"])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16], ids=["o-fp16", "o-bf16"])
def test_qbytes_mm(batch_size, input_features, input_dtype, weight_dtype, output_features, output_dtype, device):
    if device.type == "mps" and weight_dtype.is_floating_point:
        pytest.skip("Float8 types are not supported on MPS device")
    input_shape = (32, input_features)
    if batch_size is not None:
        input_shape = (batch_size,) + input_shape
    if input_dtype is None:
        input_dtype = output_dtype
    input = random_tensor(input_shape, dtype=input_dtype, device=device)
    weight = random_tensor((output_features, input_features), dtype=weight_dtype, device=device)
    # Use a scale small enough to prevent overflows
    scale = random_tensor((output_features,), dtype=output_dtype, device=device) / 1e3
    output = torch.ops.quanto.qbytes_mm(input, weight, scale)
    expected = torch.matmul(input.to(scale.dtype), weight.to(scale.dtype).t() * scale)
    assert_similar(expected, output)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA device >= sm80 not available",
)
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
