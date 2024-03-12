import pytest
import torch
from helpers import device_eq

from quanto import PackedTensor


@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
def test_packed_tensor_transpose(bits, device):
    qmax = 2**bits
    shape = (10, 32)
    unpacked = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    packed = PackedTensor.pack(unpacked, bits=bits)
    transposed = packed.t()
    assert isinstance(transposed, PackedTensor)
    assert device_eq(packed.device, device)
    assert transposed._axis == 1
    assert transposed.shape == unpacked.t().shape
    assert transposed.stride() == unpacked.t().stride()
    assert torch.equal(transposed.unpack(), unpacked.t())
