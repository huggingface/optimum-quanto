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

from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight


# Each row is a quantization vector under test along the selected axis.
PATTERNS = torch.tensor(
    [
        [1, 2, 1, 2],
        [-2, -1, -2, -1],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
    ],
    dtype=torch.float32,
)


def quantize_dequantize(base, qtype, axis, group_size):
    scale, shift = MaxOptimizer()(base, qtype=qtype, axis=axis, group_size=group_size, zeropoint=True)
    return quantize_weight(
        base,
        qtype=qtype,
        axis=axis,
        scale=scale,
        shift=shift,
        group_size=group_size,
        optimized=False,
    ).dequantize()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [None, 2], ids=["channel-wise", "group-wise"])
def test_affine_optimizer_zeropoint_contains_zero(dtype, qtype, axis, group_size, device):
    patterns = PATTERNS.to(dtype=dtype, device=device)
    # For axis=-1, each column is an independently quantized vector.
    base = patterns if axis == 0 else patterns.transpose(0, 1)
    qa = quantize_dequantize(base, qtype, axis, group_size)
    vectors = qa if axis == 0 else qa.transpose(0, 1)

    for original, dequantized in zip(patterns, vectors):
        if torch.all(original == 0):
            # All-zero intervals must stay finite and exactly zero.
            assert torch.equal(dequantized, torch.zeros_like(dequantized))
        elif torch.all(original == original[0]):
            # Non-zero constant intervals must not collapse to zero.
            assert torch.allclose(dequantized, original, atol=2e-2, rtol=2e-2)
        else:
            # One-sided intervals must keep their two distinct magnitudes instead of
            # saturating to a single value at the zero-containing boundary.
            assert torch.unique(dequantized).numel() >= 2
            if torch.all(original > 0):
                assert torch.all(dequantized > 0)
                assert torch.isclose(dequantized.max(), original.max(), atol=2e-2, rtol=2e-2)
            else:
                assert torch.all(dequantized < 0)
                assert torch.isclose(dequantized.min(), original.min(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_affine_optimizer_zeropoint_preserves_mixed_sign_intervals(qtype, axis, device):
    base = torch.tensor([[-1, 2], [2, -1]], dtype=torch.float32, device=device)
    dequantized = quantize_dequantize(base, qtype, axis, None)
    torch.testing.assert_close(dequantized, base, atol=0, rtol=0)


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_affine_optimizer_zeropoint_extreme_fp16(qtype, axis, device):
    # These one-sided intervals expose the float16 overflow of (qmax * scale - shift) / qmax.
    patterns = torch.tensor([[32752, 65504], [-65504, -32752]], dtype=torch.float16, device=device)
    base = patterns if axis == 0 else patterns.transpose(0, 1)
    scale, shift = MaxOptimizer()(base, qtype=qtype, axis=axis, group_size=None, zeropoint=True)
    assert torch.isfinite(scale).all()
    qa = quantize_weight(
        base,
        qtype=qtype,
        axis=axis,
        scale=scale,
        shift=shift,
        group_size=None,
        optimized=False,
    ).dequantize()
    vectors = qa if axis == 0 else qa.transpose(0, 1)

    for original, dequantized in zip(patterns, vectors):
        assert torch.isfinite(dequantized).all()
        torch.testing.assert_close(dequantized, original, atol=0, rtol=0.35)
        assert torch.unique(dequantized).numel() >= 2
        if torch.all(original > 0):
            assert torch.all(dequantized > 0)
        else:
            assert torch.all(dequantized < 0)
