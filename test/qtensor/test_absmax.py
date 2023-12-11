import pytest
import torch
from helpers import random_tensor

from quanto.quantization import absmax_scale


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (2, 10), (10, 32, 32)])
@pytest.mark.parametrize(
    "itype", [torch.int8, torch.float8_e5m2, torch.float8_e4m3fn], ids=["int8", "float8_e5m2", "float8_e4m3"]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_quantize_scale(input_shape, axis, dtype, itype, device):
    if device.type == "mps" and itype.is_floating_point:
        pytest.skip("Float8 are not supported on MPS device")
    a = random_tensor(input_shape, dtype=dtype).to(device)
    scale = absmax_scale(a, itype, axis)
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
