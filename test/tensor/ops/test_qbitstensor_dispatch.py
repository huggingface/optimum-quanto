from helpers import random_qbitstensor

from quanto import QBitsTensor, qint4


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
