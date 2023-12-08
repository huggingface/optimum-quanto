import pytest
import torch
from helpers import q_assert_close, random_qtensor


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
