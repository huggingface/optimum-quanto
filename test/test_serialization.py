import io

import pytest
import torch
from helpers import random_qtensor, random_tensor

from quanto.quantization import QTensor, absmax_scale
from quanto.quantization.nn import QLinear


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (2, 10), (10, 32, 32)])
@pytest.mark.parametrize("int_dtype", [torch.int8], ids=["int8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_quantized_tensor_serialization(input_shape, int_dtype, dtype, axis):
    inputs = random_tensor(input_shape, dtype=dtype)
    scale = absmax_scale(inputs, int_dtype, axis)
    qinputs = QTensor.quantize(inputs, int_dtype, scale)
    b = io.BytesIO()
    torch.save(qinputs, b)
    b.seek(0)
    qinputs_reloaded = torch.load(b)
    assert torch.equal(qinputs_reloaded._data, qinputs._data)
    assert torch.equal(qinputs_reloaded._scale, qinputs._scale)
    # We cannot test dtype directly, as it is not set correctly by torch.load
    assert qinputs_reloaded._scale.dtype == dtype
    assert qinputs_reloaded.axis == qinputs.axis


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_quantized_module_serialization(dtype, device):
    embeddings = 10
    linear = torch.nn.Linear(embeddings, embeddings).to(dtype).to(device)
    linear.to(dtype)
    qlinear = QLinear.from_module(linear)
    qlinear.freeze()
    b = io.BytesIO()
    torch.save(qlinear.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b)
    qlinear_reloaded = QLinear(embeddings, embeddings)
    # We need to force assignment instead of copy to replace weights by quantized weights
    qlinear_reloaded.load_state_dict(state_dict, assign=True)
    for attr in ["weight", "bias"]:
        t = getattr(qlinear, attr)
        t_reloaded = getattr(qlinear_reloaded, attr)
        assert torch.equal(t._data, t_reloaded._data)
        assert torch.equal(t._scale, t_reloaded._scale)
        assert t_reloaded.dtype == dtype
