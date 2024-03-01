import pytest
import torch
from helpers import random_tensor

from quanto import absmax_scale, qfloat8, qint8


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (2, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_absmax_scale(input_shape, axis, dtype, qtype, device):
    if device.type == "mps" and qtype.is_floating_point:
        pytest.skip("Float8 are not supported on MPS device")
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, qtype, axis)
    assert scale.dtype == dtype
    if axis is None:
        assert scale.ndim == 0
    else:
        assert scale.ndim == a.ndim
        sscale = torch.squeeze(scale)
        if a.ndim == 1 or a.shape[axis] == 1:
            # Quantization is actually per-tensor as the axis dim is 1
            assert sscale.ndim == 0
        else:
            assert sscale.ndim == 1


@pytest.mark.parametrize("input_shape", [(256, 256), (256, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("group_size", [64, 128])
def test_absmax_scale_groupwise(input_shape, dtype, qtype, axis, group_size, device):
    if device.type == "mps" and qtype.is_floating_point:
        pytest.skip("Float8 is not supported on MPS device")
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, qtype, axis, group_size)
    assert scale.dtype == dtype
    assert scale.shape[axis] == a.numel() // group_size
