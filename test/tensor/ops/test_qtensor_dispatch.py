import pytest
import torch
from helpers import assert_similar, random_qtensor, random_tensor

from quanto import QTensor


def test_to_device(device):
    qa = random_qtensor((32, 32), dtype=torch.float)
    qa = qa.to(device)
    assert isinstance(qa, QTensor)
    assert qa.device.type == device.type
    assert qa._data.device.type == device.type
    assert qa._scale.device.type == device.type


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("scalar", [1, 0.5, torch.tensor(0.12)], ids=["int", "float", "tensor"])
def test_mul_scalar(input_shape, scalar, device):
    qa = random_qtensor(input_shape, dtype=torch.float32).to(device)
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.to(device)
    qprod = qa * scalar
    assert isinstance(qprod, QTensor)
    prod = qa.dequantize() * scalar
    assert_similar(prod, qprod)
    qprod = scalar * qa
    assert isinstance(qprod, QTensor)
    prod = scalar * qa.dequantize()
    assert_similar(prod, qprod)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
def test_relu(batch_size, tokens, embeddings, device):
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    qout = torch.nn.functional.relu(qinputs)
    assert isinstance(qout, QTensor)
    assert torch.equal(qout._data, torch.maximum(qinputs._data, torch.zeros((1,)).to(device)))


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
def test_softmax(batch_size, tokens, embeddings, device):
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    qout = torch.nn.functional.softmax(qinputs, dim=-1)
    assert isinstance(qout, QTensor)
    assert torch.min(qout.dequantize()) >= 0
    assert torch.max(qout.dequantize()) <= 1


@pytest.mark.parametrize("input_shape", [(10,), (10, 32)])
def test_view(input_shape, device):
    qinputs = random_qtensor(input_shape, dtype=torch.float32).to(device)
    qview = qinputs.view((1,) + input_shape)
    assert isinstance(qview, QTensor)


@pytest.mark.parametrize("input_shape", [(10,), (10, 32)])
def test_cat(input_shape, device):
    qinputs = random_qtensor(input_shape, dtype=torch.float32).to(device)
    other = random_tensor(input_shape, dtype=torch.float32).to(device)
    # First, quantize other with the same scale
    qother = QTensor.quantize(other, qtype=qinputs.qtype, axis=None, group_size=None, scale=qinputs._scale)
    qcat = torch.cat([qinputs, qother])
    assert isinstance(qcat, QTensor)
    assert_similar(torch.cat([qinputs.dequantize(), qother.dequantize()]), qcat)
    # Now, verify that with different scales, the output is dequantized
    qother = QTensor.quantize(other, qinputs.qtype)
    qcat = torch.cat([qinputs, qother])
    assert not isinstance(qcat, QTensor)


@pytest.mark.parametrize(
    "axis, group_size",
    [[None, None], [0, None], [0, 2], [-1, None], [-1, 2]],
    ids=["per-tensor", "first-axis", "first-axis-groupwise", "last-axis", "last-axis-groupwise"],
)
def test_transpose_2d(axis, group_size, device):
    input_shape = (4, 6)
    qinputs = random_qtensor(input_shape, axis=axis, group_size=group_size).to(device)
    qtransposed = qinputs.t()
    assert qtransposed.qtype == qinputs.qtype
    if axis == -1:
        assert qtransposed.axis == 0
    elif axis == 0:
        assert qtransposed.axis == -1
    assert qtransposed.shape == input_shape[::-1]
    assert torch.equal(qtransposed.dequantize(), qinputs.dequantize().t())


def test_transpose(device):
    input_shape = (10, 32, 64)
    qinputs = random_qtensor(input_shape).to(device)
    qtransposed = torch.transpose(qinputs, 1, 2)
    assert qtransposed.qtype == qinputs.qtype
    assert torch.equal(qtransposed.dequantize(), torch.transpose(qinputs.dequantize(), 1, 2))
