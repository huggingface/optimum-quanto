import torch

from quanto.quantization import QTensor


def random_tensor(shape, dtype=torch.float32):
    # Return a random tensor between -1. and 1.
    return torch.rand(shape, dtype=dtype) * 2 - 1


def random_qtensor(shape, dtype=torch.float32):
    return QTensor.quantize(random_tensor(shape, dtype))


def q_assert_close(x: torch.Tensor, xq: QTensor):
    # Absolute error is the quantization scale
    atol = max(xq._scale, 1e-6)
    abs_error = torch.max(torch.abs(x - xq.dequantize()))
    assert abs_error <= atol, f"error {abs_error:.4f} exceeds atol {atol:.4f}"
