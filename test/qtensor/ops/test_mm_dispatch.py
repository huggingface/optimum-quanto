import pytest
import torch
from helpers import q_assert_close, random_qtensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("in_features", [5, 16, 24])
@pytest.mark.parametrize("hidden", [5, 16, 24])
@pytest.mark.parametrize("out_features", [5, 16, 24])
def test_matmul(dtype, in_features, hidden, out_features, device):
    if dtype == torch.float16 and device.type == "cpu":
        pytest.skip("Matrix multiplication is not supported for float16 on CPU.")
    qa = random_qtensor((in_features, hidden), dtype=dtype).to(device)
    qb = random_qtensor((hidden, out_features), dtype=dtype).to(device)
    qmatmul = torch.matmul(qa, qb)
    # The outputs should be almost identical if we use the dequantized inputs
    matmul = torch.matmul(qa.dequantize(), qb.dequantize())
    # We need to increase atol and rtol for float16
    atol = {torch.float32: 1e-6, torch.float16: 2e-3}[dtype]
    rtol = {torch.float32: 1e-5, torch.float16: 1e-2}[dtype]
    q_assert_close(matmul, qmatmul, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("a_shape, b_shape", [[(16, 32), (32, 24)], [(5, 10), (10, 6)]])
def test_bmm(dtype, batch_size, a_shape, b_shape, device):
    if dtype == torch.float16 and device.type == "cpu":
        pytest.skip("Matrix multiplication is not supported for float16 on CPU.")
    qa = random_qtensor((batch_size,) + a_shape, dtype=dtype).to(device)
    qb = random_qtensor((batch_size,) + b_shape, dtype=dtype).to(device)
    qbmm = torch.bmm(qa, qb)
    # The outputs should be almost identical if we use the dequantized inputs
    bmm = torch.bmm(qa.dequantize(), qb.dequantize())
    # We need to increase atol and rtol for float16
    atol = {torch.float32: 1e-6, torch.float16: 2e-3}[dtype]
    rtol = {torch.float32: 1e-5, torch.float16: 1e-2}[dtype]
    q_assert_close(bmm, qbmm, atol=atol, rtol=rtol)
