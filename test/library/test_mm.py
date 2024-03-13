from contextlib import nullcontext

import pytest
import torch
from helpers import random_tensor

from quanto.library import disable_extensions
from quanto.tensor.core import group
from quanto.tensor.packed import pack_weights


@pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
@pytest.mark.parametrize("output_features", [48, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_ext", [True, False], ids=["ext", "no-ext"])
def test_dqmm(input_shape, output_features, dtype, device, use_ext):
    input = random_tensor(input_shape, dtype=dtype).to(device)
    other = torch.randint(-127, 127, (input_shape[-1], output_features), dtype=torch.int8).to(device)
    other_scale = random_tensor((output_features,), dtype=dtype).to(device)
    context = nullcontext() if use_ext else disable_extensions()
    with context:
        output = torch.ops.quanto.dqmm(input, other, other_scale)
        expected = torch.ops.aten.mm(input, other * other_scale)
    assert torch.equal(expected, output)


# @pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
# @pytest.mark.parametrize("output_features", [48, 64])
# @pytest.mark.parametrize("bits", [2, 4])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
# @pytest.mark.parametrize("use_ext", [True, False], ids=["ext", "no-ext"])
# def test_packed_udqmm(input_shape, output_features, dtype, device, bits, use_ext):
#     qmax = 2**bits
#     input = random_tensor(input_shape, dtype=dtype).to(device)
#     weights = torch.randint(0, qmax, (input_shape[-1], output_features), dtype=torch.uint8).to(device)
#     packed_weights = pack_weights(weights, bits)

#     scale = random_tensor((output_features,), dtype=dtype).to(device)
#     zeropoint = torch.randint(
#         torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max, (input_shape[-1], output_features), dtype=torch.int8
#     ).to(device)

#     context = nullcontext() if use_ext else disable_extensions()
#     with context:
#         output = torch.ops.quanto.udqmm(
#             input,
#             packed_weights,
#             scale,
#             zeropoint,
#             axis=0,
#             bits=bits,
#             orig_shape=weights.shape,
#             unpacked_shape=weights.shape,
#         )

#         unpacked_weights = torch.ops.quanto.unpack(packed_weights, bits)
#         # TODO: We should probably combine it with unpack
#         unpacked_weights = unpacked_weights[: weights.shape[0]]
#         expected = torch.ops.aten.mm(input, (unpacked_weights.to(torch.int8) - zeropoint.to(torch.int8)) * scale)
#     assert torch.equal(expected, output)


# @pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
# @pytest.mark.parametrize("output_features", [48, 64])
# @pytest.mark.parametrize("bits", [2, 4])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
# @pytest.mark.parametrize("use_ext", [True, False], ids=["ext", "no-ext"])
# def test_grouped_udqmm(input_shape, output_features, dtype, device, bits, use_ext):
#     input = random_tensor(input_shape, dtype=dtype).to(device)
#     qmax = 2**bits
#     weights = torch.randint(0, qmax, (input_shape[-1], output_features), dtype=torch.uint8).to(device)

#     grouped_weights = group(weights, axis=0, group_size=int(input_shape[-1] / 4))
#     scale = random_tensor((1, grouped_weights.shape[1]), dtype=dtype).to(device)
#     zeropoint = torch.randint(
#         torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max, grouped_weights.shape, dtype=torch.int8
#     ).to(device)

#     packed_weights = pack_weights(grouped_weights, bits)

#     context = nullcontext() if use_ext else disable_extensions()
#     with context:
#         output = torch.ops.quanto.udqmm(
#             input,
#             packed_weights,
#             scale,
#             zeropoint,
#             axis=0,
#             bits=bits,
#             orig_shape=weights.shape,
#             unpacked_shape=grouped_weights.shape,
#         )
#         unpacked_weights = torch.ops.quanto.unpack(packed_weights, bits)
#         # TODO: We should probably combine it with unpack
#         unpacked_weights = unpacked_weights[: grouped_weights.shape[0]]
#         ungrouped_weights = torch.ops.quanto.ungroup(
#             (unpacked_weights.to(torch.int8) - zeropoint.to(torch.int8)) * scale,
#             axis=0,
#             orig_shape=weights.shape,
#         )
#         expected = torch.ops.aten.mm(input, ungrouped_weights)
#     assert torch.equal(expected, output)


@pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
@pytest.mark.parametrize("output_features", [48, 64])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_combined_unpack_mm_kernel(input_shape, output_features, dtype, device, bits):
    input = random_tensor(input_shape, dtype=dtype).to(device)

    qmax = 2**bits
    a = torch.randint(0, qmax, (input_shape[-1], output_features), dtype=torch.uint8).to(device)
    packed_a = pack_weights(a, bits)
    unpacked_weights = torch.ops.quanto.unpack(packed_a, bits)

    other_scale = random_tensor((output_features,), dtype=dtype).to(device)
    output = torch.ops.quanto.combined_unpack_mm_kernel(input, packed_a, other_scale, bits)

    """
    This is a naive MM operation
    This baseline helps us know we are 
    on track
    """
    ar,ac = input.shape 
    br,bc = (unpacked_weights * other_scale).shape
    expected = torch.zeros(ar, bc, dtype=input.dtype)

    def naive_mm(a, b, result):
        for i in range(ar):         
            for j in range(bc):     
                for k in range(ac): 
                    result[i,j] += a[i,k] * b[k,j]


    naive_mm(input, (unpacked_weights * other_scale), expected)

    assert torch.equal(expected, output)
