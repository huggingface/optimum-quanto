import pytest
import torch
from helpers import random_tensor, torch_min_version

from quanto import QTensor, absmax_scale, qint8


def compile_for_device(f, device):
    # Remove any side-effects form previous compilation
    torch.compiler.reset()
    # Inductor relies on Triton for inference which does not support MPS
    backend = "aot_eager" if device == torch.device("mps") else "aot_eager"
    return torch.compile(f, backend=backend)


@torch_min_version("2.3.0")
@pytest.mark.parametrize("input_shape", [(2, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8], ids=["qint8"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
def test_compile_quantize_tensor(input_shape, qtype, axis, dtype, device):
    if device == torch.device("mps") and dtype == torch.bfloat16:
        pytest.skip("BFloat16 is not supported on MPS")
    a = random_tensor(input_shape, dtype=dtype).to(device)

    def f(x, qtype, axis):
        scale = None if axis is None else absmax_scale(a, qtype, axis).to(device)
        return QTensor.quantize(x, qtype, scale)

    compiled_f = compile_for_device(f, device)
    qa = compiled_f(a, qtype, axis)
    assert isinstance(qa, QTensor)
    assert qa.qtype == qtype
    assert qa._scale.dtype == dtype
    expected_axis = a.ndim - 1 if axis == -1 else axis
    assert qa.axis == expected_axis


@torch_min_version("2.3.0")
@pytest.mark.parametrize("qtensor_input", [True, False], ids=["qtensor-input", "tensor-input"])
def test_compile_qtensor_to(qtensor_input, device):
    input_shape = (10, 32, 32)
    a = random_tensor(input_shape).to(device)

    def f(x, dtype):
        qx = x if isinstance(x, QTensor) else QTensor.quantize(x)
        return qx.to(dtype)

    compiled_f = compile_for_device(f, device)

    if qtensor_input:
        a = QTensor.quantize(a)
    qa = compiled_f(a, torch.float16)
    assert isinstance(qa, QTensor)
    assert qa.qtype == qint8
    assert qa._scale.dtype == torch.float16
