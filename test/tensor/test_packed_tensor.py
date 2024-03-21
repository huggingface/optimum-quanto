import io

import pytest
import torch
from helpers import device_eq

from quanto import PackedTensor


@pytest.mark.parametrize("shape", [(10,), (12,), (10, 10), (12, 10), (32, 32)])
@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
def test_pack_tensor(shape, bits, device):
    """This test verifies that an integer tensor in the correct range is preserved."""
    qmax = 2 ** bits
    t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    packed = PackedTensor.pack(t, bits=bits)

    assert isinstance(packed, PackedTensor)
    assert packed.dtype == torch.uint8
    assert device_eq(packed.device, device)
    assert torch.equal(t, packed.unpack())


@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
def test_packed_tensor_serialization(bits, device):
    qmax = 2 ** bits
    shape = (10, 32)
    t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    packed = PackedTensor.pack(t, bits=bits)
    b = io.BytesIO()
    torch.save(packed, b)
    b.seek(0)
    packed_reloaded = torch.load(b)
    assert isinstance(packed_reloaded, PackedTensor)
    assert packed_reloaded.shape == packed.shape
    assert packed_reloaded.dtype == packed.dtype
    assert packed_reloaded.bits == packed.bits
    assert torch.equal(packed_reloaded._data, packed._data)
    assert torch.equal(t, packed_reloaded.unpack())
