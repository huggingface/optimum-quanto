import pytest
import torch

from quanto.tensor.core import int2, int4, pack_weights


@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
@pytest.mark.parametrize("shape", [(12,), (32, 32)], ids=["vector", "matrix"])
def test_unpack(bits, shape):
    qmax = 2**bits
    a = torch.randint(0, qmax, shape, dtype=torch.uint8)
    bitsdtype = int2 if bits == 2 else int4
    packed_a = pack_weights(a, bitsdtype)
    unpacked_a = torch.ops.quanto.unpack(packed_a, bits)
    assert torch.equal(unpacked_a, a)
