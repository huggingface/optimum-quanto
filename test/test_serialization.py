import io

import pytest
import torch
from helpers import random_qtensor, random_tensor

from quanto.quantization import QTensor, absmax_scale, calibration
from quanto.quantization.nn import QLinear


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (2, 10), (10, 32, 32)])
@pytest.mark.parametrize("itype", [torch.int8], ids=["int8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_quantized_tensor_serialization(input_shape, itype, dtype, axis):
    inputs = random_tensor(input_shape, dtype=dtype)
    scale = absmax_scale(inputs, itype, axis)
    qinputs = QTensor.quantize(inputs, itype, scale)
    b = io.BytesIO()
    torch.save(qinputs, b)
    b.seek(0)
    qinputs_reloaded = torch.load(b)
    assert torch.equal(qinputs_reloaded._data, qinputs._data)
    assert torch.equal(qinputs_reloaded._scale, qinputs._scale)
    # We cannot test dtype directly, as it is not set correctly by torch.load
    assert qinputs_reloaded._scale.dtype == dtype
    assert qinputs_reloaded.axis == qinputs.axis


@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize(
    "dtype, per_axis",
    [[torch.float16, None], [torch.float32, False], [torch.float32, True]],
    ids=["fp16", "fp32-per-tensor", "fp32-per-axis"],
)
def test_quantized_module_serialization(use_bias, dtype, per_axis, device):
    embeddings = 10
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(dtype).to(device)
    linear.to(dtype)
    qlinear = QLinear.from_module(linear)
    if per_axis is not None:
        qinputs = random_qtensor((10, 10, embeddings), dtype=dtype).to(device)
        with calibration(per_axis=per_axis):
            qlinear(qinputs)
    qlinear.freeze()
    b = io.BytesIO()
    torch.save(qlinear.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b)
    qlinear_reloaded = QLinear(embeddings, embeddings, bias=use_bias)
    # We need to force assignment instead of copy to replace weights by quantized weights
    qlinear_reloaded.load_state_dict(state_dict, assign=True)
    w = qlinear.weight
    w_reloaded = qlinear_reloaded.weight
    assert torch.equal(w._data, w_reloaded._data)
    assert torch.equal(w._scale, w_reloaded._scale)
    assert w_reloaded.dtype == dtype
    assert w_reloaded.axis == w.axis
    if per_axis is not None:
        for attr in ["input", "output"]:
            v = getattr(qlinear.scales, attr)
            assert v is not None
            v_reloaded = getattr(qlinear_reloaded.scales, attr)
            assert torch.equal(v, v_reloaded)
