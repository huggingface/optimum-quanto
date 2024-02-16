from contextlib import nullcontext

import pytest
import torch

from quanto.library import disable_extensions
from quanto.tensor.core import pack_weights, qint2, qint4


@pytest.mark.parametrize("bits", [2, 4], ids=["int2", "int4"])
@pytest.mark.parametrize("shape", [(12,), (32, 32)], ids=["vector", "matrix"])
@pytest.mark.parametrize("use_ext", [True, False], ids=["ext", "no-ext"])
def test_unpack(bits, shape, use_ext, device):
    qmax = 2**bits
    a = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    bitsdtype = qint2 if bits == 2 else qint4
    packed_a = pack_weights(a, bitsdtype)
    context = nullcontext() if use_ext else disable_extensions()
    with context:
        unpacked_a = torch.ops.quanto.unpack(packed_a, bits)
    assert torch.equal(unpacked_a, a)
