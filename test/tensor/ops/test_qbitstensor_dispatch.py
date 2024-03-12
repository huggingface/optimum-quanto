import pytest
import torch
from helpers import device_eq, random_qbitstensor, random_tensor

from quanto import QBitsTensor, qint2, qint4


def test_to_device(device):
    qa = random_qbitstensor((32, 32), qtype=qint4)
    qa = qa.to(device)
    assert isinstance(qa, QBitsTensor)
    assert qa.device.type == device.type
    assert qa._data.device.type == device.type
    assert qa._scale.device.type == device.type
    assert qa._zeropoint.device.type == device.type


def test_detach():
    qa = random_qbitstensor((32, 32), qtype=qint4)
    dqa = qa.detach()
    assert isinstance(dqa, QBitsTensor)


@pytest.mark.parametrize("qtype", [qint2, qint4], ids=["qint2", "qint4"])
@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
def test_transpose(qtype, axis, device):
    a = random_tensor((32, 64)).to(device)
    qa = QBitsTensor.quantize(a, qtype, axis)
    transposed = qa.t()
    assert isinstance(transposed, QBitsTensor)
    assert device_eq(transposed.device, device)
    assert transposed.qtype == qtype
    assert transposed.axis == (-1 if axis == 0 else 0)
    assert transposed.shape == a.t().shape
    assert transposed.stride() == a.t().stride()
    assert torch.equal(transposed.dequantize(), qa.dequantize().t())
