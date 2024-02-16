import io

import pytest
import torch
from helpers import random_qtensor, random_tensor

from quanto import Calibration, QTensor, absmax_scale, freeze, qfloat8, qint8, quantize
from quanto.nn import QLinear, QModuleMixin


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (2, 10), (10, 32, 32)])
@pytest.mark.parametrize("qtype", [qint8, qfloat8], ids=["qint8", "qfloat8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("axis", [None, 0, -1], ids=["per-tensor", "first-axis", "last-axis"])
def test_quantized_tensor_serialization(input_shape, qtype, dtype, axis):
    inputs = random_tensor(input_shape, dtype=dtype)
    scale = absmax_scale(inputs, qtype, axis)
    qinputs = QTensor.quantize(inputs, qtype, scale)
    b = io.BytesIO()
    torch.save(qinputs, b)
    b.seek(0)
    qinputs_reloaded = torch.load(b)
    assert qinputs_reloaded.qtype == qtype
    assert torch.equal(qinputs_reloaded._scale, qinputs._scale)
    if qtype.is_floating_point:
        # Equality is not supported for float8
        assert torch.equal(qinputs_reloaded._data.to(torch.float32), qinputs._data.to(torch.float32))
    else:
        assert torch.equal(qinputs_reloaded._data, qinputs._data)
    # We cannot test dtype directly as it is not correctly set by torch.load
    assert qinputs_reloaded._scale.dtype == dtype
    assert qinputs_reloaded.axis == qinputs.axis


@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("activations", [None, qint8], ids=["a-float", "a-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_qlinear_serialization(use_bias, activations, weights, dtype, device):
    if dtype == torch.float16 and device.type == "cpu":
        pytest.skip("Matrix multiplication is not supported for float16 on CPU")
    embeddings = 10
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(dtype).to(device)
    qlinear = QLinear.from_module(linear, weights=weights, activations=activations)
    if activations is not None:
        qinputs = random_qtensor((10, 10, embeddings), dtype=dtype).to(device)
        with Calibration():
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
    assert w.qtype == w_reloaded.qtype
    assert torch.equal(w._data, w_reloaded._data)
    assert torch.equal(w._scale, w_reloaded._scale)
    assert w_reloaded.dtype == dtype
    assert w_reloaded.axis == w.axis
    if activations is not None:
        for attr in ["input_scale", "output_scale"]:
            v = getattr(qlinear, attr)
            v_reloaded = getattr(qlinear_reloaded, attr)
            assert torch.equal(v, v_reloaded)


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.mid_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = torch.nn.functional.relu(self.input_layer(inputs))
        x = torch.nn.functional.relu(self.mid_layer(x))
        return torch.nn.functional.softmax(self.output_layer(x), dim=-1)


@pytest.mark.parametrize("weights", [qint8], ids=["w-qint8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_serialize_quantized_mlp(weights, dtype, device):
    if dtype == torch.float16 and device.type == "cpu":
        pytest.skip("Matrix multiplication is not supported for float16 on CPU")
    input_features = 32
    hidden_features = 10
    output_features = 128
    model = MLP(input_features, hidden_features, output_features).to(dtype).to(device)
    quantize(model, weights=weights)
    inputs = random_tensor((1, 10, input_features), dtype=dtype).to(device)
    with Calibration():
        model(inputs)
    freeze(model)
    b = io.BytesIO()
    torch.save(model.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b)
    model_reloaded = MLP(input_features, hidden_features, output_features).to(device)
    quantize(model_reloaded)
    # When reloading we must assign instead of copying to force quantized tensors assignment
    model_reloaded.load_state_dict(state_dict, assign=True)
    for name, module in model.named_modules():
        if isinstance(module, QModuleMixin):
            module_reloaded = getattr(model_reloaded, name)
            assert torch.equal(module_reloaded.weight._data, module.weight._data)
            assert torch.equal(module_reloaded.weight._scale, module.weight._scale)
            assert torch.equal(module_reloaded.input_scale, module.input_scale)
            assert torch.equal(module_reloaded.output_scale, module.output_scale)
