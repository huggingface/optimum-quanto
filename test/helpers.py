import torch

from quanto.quantization import QTensor, absmax_scale


def device_eq(a, b):
    if a.type != b.type:
        return False
    a_index = a.index if a.index is not None else 0
    b_index = b.index if b.index is not None else 0
    return a_index == b_index


def random_tensor(shape, dtype=torch.float32):
    # Return a random tensor between -1. and 1.
    return torch.rand(shape, dtype=dtype) * 2 - 1


def random_qtensor(shape, dtype=torch.float32, axis=None):
    t = random_tensor(shape, dtype)
    scale = absmax_scale(t, axis=axis)
    return QTensor.quantize(t, scale=scale)


def q_assert_close(x: torch.Tensor, xq: QTensor):
    # Absolute error is the quantization scale
    atol = torch.maximum(xq._scale, torch.tensor(1e-6))
    abs_error = torch.abs(x - xq.dequantize())
    if not torch.all(abs_error <= atol):
        print("Float", x)
        print("Quantized", xq.dequantize())
        raise ValueError("Error exceeds absolute tolerance")
