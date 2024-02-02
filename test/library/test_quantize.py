import pytest
import torch
from helpers import random_tensor


@pytest.mark.parametrize("shape", [(12,), (32, 32)], ids=["vector", "matrix"])
@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.float8_e4m3fn], ids=["int8", "float8"])
@pytest.mark.parametrize("per_axis", [True, False], ids=["per-axis", "per-tensor"])
def test_quantize_symmetric(shape, src_dtype, dst_dtype, per_axis, device):
    if device.type == "mps" and dst_dtype != torch.int8:
        pytest.skip("float8 types are not supported on MPS device")
    # Craft manually data and scale
    if dst_dtype.is_floating_point:
        data = random_tensor(shape, torch.float16).to(dst_dtype).to(device)
    else:
        data = torch.randint(-127, 127, shape, dtype=dst_dtype).to(device)
    if per_axis:
        scale_shape = (shape[0],) + (1,) * (len(shape) - 1)
    else:
        scale_shape = ()
    scale = torch.rand(scale_shape, dtype=src_dtype).to(device)
    # Dequantize to obtain a float tensor
    t = data.to(src_dtype) * scale
    qdata = torch.ops.quanto.quantize_symmetric(t, scale, dst_dtype)
    assert qdata.dtype == dst_dtype
    assert qdata.shape == shape
    # float8 tensors direct comparison is not supported yet on CPU
    if dst_dtype.is_floating_point:
        assert torch.equal(qdata.to(torch.float16), data.to(torch.float16))
    else:
        assert torch.equal(qdata, data)
