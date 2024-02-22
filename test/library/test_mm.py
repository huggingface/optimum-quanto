import pytest
import torch
from helpers import random_tensor


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
