import pytest
import torch
from helpers import random_qweight, random_tensor

from optimum.quanto import WeightQBytesTensor, qint8, quantize_weight


def test_weight_qytes_tensor_to_device(device):
    qa = random_qweight((32, 32), qtype=qint8, dtype=torch.float)
    qa = qa.to(device)
    assert isinstance(qa, WeightQBytesTensor)
    assert qa.device.type == device.type
    assert qa._data.device.type == device.type
    assert qa._scale.device.type == device.type


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("qtype", [qint8])
def test_weight_qbytes_tensor_transpose_contiguous(axis, qtype, device):
    input_shape = (16, 32)
    qa = random_qweight(input_shape, axis=axis, qtype=qtype, dtype=torch.float32).to(device)
    assert qa.is_contiguous()
    tqa = qa.t()
    assert isinstance(tqa, WeightQBytesTensor)
    assert not tqa.is_contiguous()
    tqa = tqa.contiguous()
    assert tqa.is_contiguous()


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize("qtype", [qint8])
def test_weight_qbytes_tensor_transposed_stride(axis, qtype, device):
    input_shape = (16, 32)
    a = random_tensor(input_shape, dtype=torch.float32).to(device)
    qa = quantize_weight(a, qtype=qtype, axis=axis)
    assert qa.stride() == a.stride()
    ta = a.t()
    tqa = qa.t()
    assert isinstance(tqa, WeightQBytesTensor)
    assert tqa.stride() == ta.stride()
