import pytest
import torch
from helpers import q_assert_close, random_qtensor, random_tensor

from quanto.quantization import QTensor


def test_to_device(device):
    qa = random_qtensor((32, 32), dtype=torch.float)
    qa = qa.to(device)
    assert qa.device.type == device.type


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
def test_add(input_shape, device):
    qa = random_qtensor(input_shape, dtype=torch.float32).to(device)
    # Quantized sum will have int16 data
    qsum = qa + qa
    a = qa.dequantize()
    sum = a + a
    q_assert_close(sum, qsum)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
def test_mul(input_shape, device):
    qa = random_qtensor(input_shape, dtype=torch.float32).to(device)
    qb = random_qtensor(input_shape, dtype=torch.float32).to(device)
    # Quantized product will have int32 data
    qprod = qa * qb
    prod = qa.dequantize() * qb.dequantize()
    q_assert_close(prod, qprod)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("scalar", [1, 0.5, torch.tensor(0.12)], ids=["int", "float", "tensor"])
def test_mul_scalar(input_shape, scalar, device):
    qa = random_qtensor(input_shape, dtype=torch.float32).to(device)
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.to(device)
    qprod = qa * scalar
    assert isinstance(qprod, QTensor)
    prod = qa.dequantize() * scalar
    q_assert_close(prod, qprod)
    qprod = scalar * qa
    assert isinstance(qprod, QTensor)
    prod = scalar * qa.dequantize()
    q_assert_close(prod, qprod)


@pytest.mark.parametrize("input_shape", [(10, 10), (32, 32)])
def test_matmul(input_shape, device):
    qa = random_qtensor(input_shape, dtype=torch.float32).to(device)
    qb = random_qtensor(input_shape, dtype=torch.float32).to(device)
    qmatmul = torch.matmul(qa, qb)
    # The outputs should be almost identical if we use the dequantized inputs
    matmul = torch.matmul(qa.dequantize(), qb.dequantize())
    q_assert_close(matmul, qmatmul)


@pytest.mark.parametrize("input_shape", [(1, 32, 32)])
def test_bmm(input_shape, device):
    qa = random_qtensor(input_shape, dtype=torch.float32).to(device)
    qb = random_qtensor(input_shape, dtype=torch.float32).to(device)
    qbmm = torch.bmm(qa, qb)
    # The outputs should be almost identical if we use the dequantized inputs
    bmm = torch.bmm(qa.dequantize(), qb.dequantize())
    q_assert_close(bmm, qbmm)


@pytest.mark.parametrize("input_size", [1, 10, 32])
def test_dot(input_size, device):
    qa = random_qtensor((input_size,), dtype=torch.float32).to(device)
    qb = random_qtensor((input_size,), dtype=torch.float32).to(device)
    qdot = torch.dot(qa, qb)
    # The outputs should be almost identical if we use the dequantized inputs
    dot = torch.dot(qa.dequantize(), qb.dequantize())
    q_assert_close(dot, qdot)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_linear(batch_size, tokens, embeddings, use_bias, device):
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    qweight = random_qtensor((embeddings, embeddings), dtype=torch.float32).to(device)
    if use_bias:
        bias = random_tensor((embeddings,), dtype=torch.float32).to(device)
        bias_scale = qinputs._scale * qweight._scale
        # Bias must be quantized with a higher bitwidth as they are added to the product of two int8
        qbias = QTensor.quantize(bias, torch.int32, bias_scale)
        q_assert_close(bias, qbias)
    else:
        bias = None
        qbias = None
    out = torch.nn.functional.linear(qinputs.dequantize(), qweight.dequantize(), bias)
    qout = torch.nn.functional.linear(qinputs, qweight, qbias)
    q_assert_close(out, qout)


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
    qother = QTensor.quantize(other, qinputs._data.dtype, qinputs._scale)
    qcat = torch.cat([qinputs, qother])
    assert isinstance(qcat, QTensor)
    q_assert_close(torch.cat([qinputs.dequantize(), qother.dequantize()]), qcat)
    # Now, verify that with different scales, the output is dequantized
    qother = QTensor.quantize(other, qinputs._data.dtype)
    qcat = torch.cat([qinputs, qother])
    assert not isinstance(qcat, QTensor)
