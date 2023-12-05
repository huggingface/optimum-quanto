import io

import pytest
import torch
from helpers import random_qtensor, random_tensor

from quanto.quantization import QTensor, absmax_scale


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


@pytest.mark.parametrize("quantize_before_load", [True, False])
def test_quantized_module_serialization(device, quantize_before_load):
    embeddings = 10

    def new_qlinear(embeddings, device):
        qweight = random_qtensor((embeddings, embeddings), dtype=torch.float32).to(device)
        qbias = random_qtensor((embeddings,), dtype=torch.float32).to(device)
        qlinear = torch.nn.Linear(embeddings, embeddings)
        with torch.no_grad():
            qlinear.weight = torch.nn.Parameter(qweight)
            qlinear.bias = torch.nn.Parameter(qbias)
        return qlinear

    qlinear = new_qlinear(embeddings, device)
    b = io.BytesIO()
    torch.save(qlinear.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b)
    if quantize_before_load:
        qlinear_reloaded = new_qlinear(embeddings, device)
        # Since the weights are already quantized tensors, we can copy instead of assigning
        assign = False
    else:
        qlinear_reloaded = torch.nn.Linear(embeddings, embeddings)
        # We need to force assignment instead of copy to replace weights by quantized weights
        assign = True
    qlinear_reloaded.load_state_dict(state_dict, assign=assign)
    for attr in ["weight", "bias"]:
        t = getattr(qlinear, attr)
        t_reloaded = getattr(qlinear_reloaded, attr)
        assert torch.equal(t._data, t_reloaded._data)
        assert torch.equal(t._scale, t_reloaded._scale)
