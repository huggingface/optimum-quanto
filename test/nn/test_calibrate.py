import pytest
import torch
from helpers import q_assert_close, random_qtensor

from quanto.quantization import QTensor, calibration, freeze
from quanto.quantization.nn import QLinear


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("per_axis", [True, False], ids=["per-axis", "per-tensor"])
def test_calibrate_qlinear(batch_size, tokens, embeddings, use_bias, per_axis, device):
    linear = torch.nn.Linear(embeddings, embeddings, bias=use_bias).to(device)
    qlinear = QLinear.from_module(linear)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    # Run a first inference without calibration
    with torch.no_grad():
        qout = qlinear(qinputs)
    assert qout.dtype == qinputs.dtype
    if not use_bias:
        assert isinstance(qout, QTensor)
        assert qout.itype == torch.int32
    else:
        assert not isinstance(qout, QTensor)
    assert qlinear.scales.input is None
    assert qlinear.scales.output is None
    # Calibrate to adjust input and output scales
    with torch.no_grad(), calibration(per_axis=per_axis):
        qout = qlinear(qinputs)
    assert isinstance(qout, QTensor)
    assert qout.dtype == qinputs.dtype
    assert qout.itype == torch.int8
    assert qlinear.scales.input is not None
    assert qlinear.scales.output is not None
    if per_axis:
        assert qout.axis == 2
    # Freeze to set quantized weights
    freeze(qlinear)
    # Align linear weights with quantized linear weights for comparison
    linear.weight = torch.nn.Parameter(qlinear.weight.dequantize())
    with torch.no_grad():
        out = linear(qinputs.dequantize())
    q_assert_close(out, qout)
    # Now run an inference without calibrating
    with torch.no_grad():
        int_qout = qlinear(qinputs)
    assert torch.equal(qout._scale, int_qout._scale)
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1


@pytest.mark.parametrize("per_axis", [True, False], ids=["per-axis", "per-tensor"])
def test_calibrate_custom_module(per_axis):
    tokens = 10
    embeddings = 32

    class TwoLinearModel(torch.nn.Module):
        def __init__(self, embeddings):
            super().__init__()
            self.linear1 = torch.nn.Linear(embeddings, embeddings)
            self.linear2 = torch.nn.Linear(embeddings, embeddings)

        def forward(self, input):
            return self.linear2(self.linear1(input))

    model = TwoLinearModel(embeddings)
    model.linear1 = QLinear.from_module(model.linear1)
    model.linear2 = QLinear.from_module(model.linear2)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32)
    with torch.no_grad(), calibration(per_axis=per_axis):
        qout = model(qinputs)
    assert model.linear1.scales.input is not None
    assert model.linear1.scales.output is not None
    assert model.linear2.scales.input is not None
    assert model.linear2.scales.output is not None
    if not per_axis:
        assert isinstance(qout, QTensor)
        assert qout.itype == torch.int8
