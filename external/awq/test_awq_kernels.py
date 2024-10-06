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
from pack import pack_awq

from optimum.quanto import AffineQuantizer, MaxOptimizer, qint4, ungroup


def assert_similar(a, b, atol=None, rtol=None):
    """Verify that the cosine similarity of the two inputs is close to 1.0 everywhere"""
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    if atol is None:
        # We use torch finfo resolution
        atol = torch.finfo(a.dtype).resolution
    if rtol is None:
        # Please refer to that discussion for default rtol values based on the float type:
        # https://scicomp.stackexchange.com/questions/43111/float-equality-tolerance-for-single-and-half-precision
        rtol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1e-1}[a.dtype]
    sim = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    if not torch.allclose(sim, torch.tensor(1.0, dtype=sim.dtype), atol=atol, rtol=rtol):
        max_deviation = torch.min(sim)
        raise ValueError(f"Alignment {max_deviation:.8f} deviates too much from 1.0 with atol={atol}, rtol={rtol}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features, out_features", [(256, 256), (512, 256)])
@pytest.mark.parametrize("kernel", ["gemv", "gemm"])
def test_standalone_kernel(in_features, out_features, kernel):
    """This test verifies that the GEMM operation is equivalent to torch.mm.
    """
    bits = 4
    group_size = 128 # Hard-coded in kernels
    interleave = 4 # Hard-coded in kernels
    kstride = 64 # Hard-coded in kernels
    device = torch.device('cuda')
    batch_size, tokens = (4, 1) if kernel =="gemv" else (10, 128)
    input_shape = (batch_size, tokens, in_features)
    # FIXME: does not work if inputs are negative !!??
    inputs = torch.rand(input_shape, dtype=torch.float16, device=device)
    qmax = 2**bits
    other_shape = (out_features, in_features)
    other_data = torch.randint(0, qmax, other_shape, dtype=torch.uint8, device=device)
    #packed_other_data = pack_intweight(other_data.to(torch.int32), interleave=interleave, kstride=kstride)
    packed_other_data = pack_awq(other_data.to(torch.int32), interleave=interleave, kstride=kstride)
    # The GEMM kernel works on transposed scales
    scales_shape = (in_features // group_size, out_features)
    other_scales = torch.rand(scales_shape, dtype=torch.float16, device=device) / qmax
    # The GEMM kernel works on transposed, negated and scaled zeropoints
    qmin = -2**(bits -1)
    qmax = 2**(bits -1)
    other_zeropoints = torch.randint(qmin, qmax, scales_shape, dtype=torch.int8, device=device)
    # Negate and scale
    other_scaled_zeropoints = - other_zeropoints * other_scales
    # Evaluate mm outputs using the GEMM kernel
    if kernel == "gemv":
        awq_outputs = torch.ops.quanto.gemv(inputs,
                                         packed_other_data,
                                         other_scales,
                                         other_scaled_zeropoints,
                                         rows=inputs.numel() // inputs.shape[-1],
                                         out_cols=out_features,
                                         in_cols=in_features,
                                         bits=4,
                                         group_size=group_size)
    else:
        awq_outputs = torch.ops.quanto.gemm(inputs,
                                                  packed_other_data,
                                                  other_scales,
                                                  other_scaled_zeropoints,
                                                  rows=inputs.numel() // inputs.shape[-1],
                                                  out_cols=out_features,
                                                  in_cols=in_features,
                                                  bits=4,
                                                  group_size=group_size)
    # Transpose other data and reshape it to align it with transposed scales and zeros
    other_data_t = other_data.t().reshape(group_size, in_features // group_size, out_features)
    # Dequantize transposed other
    other_t = (other_data_t - other_zeropoints) * other_scales
    # Reshape it as expected by the matmul
    other_t = other_t.reshape(in_features, out_features)
    # Evaluate the matrix multiplication using pytorch float16 mm
    pt_outputs = torch.matmul(inputs, other_t)
    # Verify the results are similar
    assert_similar(awq_outputs, pt_outputs, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features, out_features", [(256, 256), (512, 256)])
@pytest.mark.parametrize("kernel", ["gemm", "gemv"])
def test_integrated_kernel(in_features, out_features, kernel):
    group_size = 128 # Hard-coded in kernels
    interleave = 4 # Hard-coded in kernels
    kstride = 64 # Hard-coded in kernels
    device = torch.device('cuda')
    batch_size, tokens = (4, 1) if kernel == "gemv" else (10, 128)
    input_shape = (batch_size, tokens, in_features)
    inputs = torch.rand(input_shape, dtype=torch.float16, device=device) * 2 - 1
    other_shape = (out_features, in_features)
    other = torch.rand(other_shape, dtype=torch.float16, device=device) * 2 - 1
    # Quantize using quanto
    scale, zeropoint = MaxOptimizer()(other, bits=4, axis=0, group_size=128)
    quanto_base = AffineQuantizer.apply(other, qint4, 0, group_size, scale, zeropoint)
    # Evaluate mm
    quanto_outputs = torch.matmul(inputs, quanto_base.t())

    # Extract quantized data, unpack and ungroup to recover original shape
    quanto_data = ungroup(quanto_base._data.unpack(), axis=0, orig_shape=other_shape)
    # Pack data for AWQ kernel
    awq_data = pack_awq(quanto_data.to(torch.int32), interleave=interleave, kstride=kstride)
    # Reshape and transpose scale as expected by AWQ kernel (! buffer must be contiguous)
    awq_scale = scale.reshape(out_features, in_features // group_size).t().contiguous()
    # Reshape and transpose zeropoint as expected by AWQ kernel (! buffer must be contiguous)
    awq_zeropoint = zeropoint.reshape(out_features, in_features // group_size).t().contiguous()
    # Negate and rescale
    awq_scaled_zeropoint = - awq_zeropoint * awq_scale

    # Evaluate mm outputs using the AWQ kernels
    if kernel == "gemv":
        awq_outputs = torch.ops.quanto.gemv(inputs,
                                         awq_data,
                                         awq_scale,
                                         awq_scaled_zeropoint,
                                         rows=inputs.numel() // inputs.shape[-1],
                                         out_cols=out_features,
                                         in_cols=in_features,
                                         bits=4,
                                         group_size=group_size)
    else:
        awq_outputs = torch.ops.quanto.gemm(inputs,
                                                  awq_data,
                                                  awq_scale,
                                                  awq_scaled_zeropoint,
                                                  rows=inputs.numel() // inputs.shape[-1],
                                                  out_cols=out_features,
                                                  in_cols=in_features,
                                                  bits=4,
                                                  group_size=group_size)

    # Verify the results are similar
    assert_similar(awq_outputs, quanto_outputs, rtol=5e-3)
