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


def q_assert_close(x: torch.Tensor, xq: QTensor, atol: float = None, rtol: float = None):
    # Please refer to that discussion for default rtol values based on the float type:
    # https://scicomp.stackexchange.com/questions/43111/float-equality-tolerance-for-single-and-half-precision
    dtype = x.dtype
    if atol is None:
        # We use torch finfo resolution
        atol = torch.finfo(x.dtype).resolution
    # We cannot expect an absolute error lower than the quantization scale
    atol = torch.maximum(xq._scale, torch.tensor(atol))
    if rtol is None:
        # Please refer to that discussion for default rtol values based on the float type:
        # https://scicomp.stackexchange.com/questions/43111/float-equality-tolerance-for-single-and-half-precision
        rtol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1e-1}[dtype]
    xdq = xq.dequantize()
    abs_error = torch.abs(x - xdq)
    closeness = atol + rtol * torch.abs(x)
    if not torch.all(abs_error <= closeness):
        max_error_index = torch.argmax(abs_error - closeness)
        max_error_x = x.flatten()[max_error_index]
        max_rel_error = abs_error.flatten()[max_error_index] / torch.abs(max_error_x) * 100
        max_error_xdq = xdq.flatten()[max_error_index]
        raise ValueError(
            f"Error exceeds tolerance (max: {max_error_xdq:.8f} instead of {max_error_x:.8f} ({max_rel_error:.4f} %)."
        )
