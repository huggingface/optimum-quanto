import pytest
import torch

from optimum.quanto import AffineQuantizer, MaxOptimizer, qint4, ungroup


def awq_quantize(base, scales, zeros, group_size):
    _, in_features = base.shape
    scale_zeros = scales * zeros
    intweight = []
    # From https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemv_fast.py#L165
    for idx in range(in_features):
        intweight.append(
            torch.round(
                (base[:, idx] + scale_zeros[:, idx // group_size])
                        / scales[:, idx // group_size]
                    ).to(torch.uint8)[:, None]
                )
    intweight = torch.cat(intweight, dim=1)
    return intweight


@pytest.mark.parametrize("in_features, out_features", [(256, 512), (1024, 1024)])
def test_awq_quantize(in_features, out_features):
    """Verify that AWQ quantization is equivalent to quanto affine quantization
    """
    shape = (out_features, in_features)
    base = torch.rand(shape, dtype=torch.float16)
    group_size = 128

    # Quantize using quanto
    scale, zeropoint = MaxOptimizer()(base, bits=4, axis=0, group_size=128)
    quanto_base = AffineQuantizer.apply(base, qint4, 0, group_size, scale, zeropoint)
    # Extract quantized data, unpack and ungroup to recover original shape
    quanto_data = ungroup(quanto_base._data.unpack(), axis=0, orig_shape=shape)

    # Reshape scale and zeropoint as expected by awq
    awq_shape = (out_features, in_features // group_size)
    scale = scale.reshape(awq_shape)
    zeropoint = zeropoint.reshape(awq_shape)

    # Compare with awq quantization
    awq_data = awq_quantize(base, scale, zeropoint, group_size)
    # FIX: AWQ does not clamp values before packing
    qmax = 2 ** 4 - 1
    awq_data = torch.clamp(awq_data, 0, qmax)

    mismatches = quanto_data != awq_data
    n = torch.sum(mismatches).numpy()
    rate = n / base.numel()
    print(f"Mismatches: {n}/{base.numel()} ({rate:.8f} %)")
    # Extract mismatches
    display = 10
    quanto_values = torch.masked_select(quanto_data, mismatches)[:display]
    awq_values = torch.masked_select(awq_data, mismatches)[:display]
    print(f"First {display} mismatches")
    print(list(quanto_values.numpy()))
    print(list(awq_values.numpy()))
    # Due to a slightly different order of operations (zero is multiplied by scale before subtracting it),
    # there are some mismatches
    assert rate < 5e-4
