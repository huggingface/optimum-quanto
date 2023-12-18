import os
from tempfile import TemporaryDirectory

import pytest
import torch
from helpers import q_assert_close, random_qtensor

from quanto import Calibration, QModuleMixin, QTensor, freeze, register_qmodule


class Conv1D(torch.nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = torch.nn.Parameter(torch.empty(nx, nf))
        self.bias = torch.nn.Parameter(torch.zeros(nf))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


@register_qmodule(Conv1D)
class QConv1D(QModuleMixin, Conv1D):
    @classmethod
    def from_module(cls, module):
        nx, nf = module.weight.size()
        qmodule = cls(nf, nx)
        with torch.no_grad():
            qmodule.weight.copy_(module.weight)
            qmodule.bias.copy_(module.bias)
        return qmodule.to(module.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If needed, quantize inputs, weights and bias
        if isinstance(input, QTensor):
            if input.itype == torch.int32:
                # Reduce input bitwidth
                input = input.rescale(torch.int8, self.in_scale)
        else:
            input = QTensor.quantize(input, torch.int8, self.in_scale)
        weight = self.weight
        if not isinstance(weight, QTensor):
            weight = QTensor.quantize(weight)
        bias = self.bias
        bias_scale = self.in_scale * weight._scale
        if isinstance(bias, QTensor):
            if bias._scale != bias_scale:
                # This should only happen if we calibrate again a frozen module
                bias = QTensor.rescale(torch.int32, bias_scale)
        else:
            bias = QTensor.quantize(bias, torch.int32, bias_scale)
        # Operate on quantized tensors
        size_out = input.size()[:-1] + (self.nf,)
        out_int32 = torch.addmm(bias, input.view(-1, input.size(-1)), weight)
        out_int32 = out_int32.view(size_out)
        # Downscale
        return out_int32.rescale(torch.int8, self.out_scale)

    def freeze(self):
        # Replace float weights by quantized weights
        self.weight = torch.nn.Parameter(QTensor.quantize(self.weight).to(self.weight.device))
        bias_scale = self.in_scale * self.weight._scale
        self.bias = torch.nn.Parameter(QTensor.quantize(self.bias, torch.int32, bias_scale))


@pytest.mark.skip("QConv1D does not work")
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
def test_quantize_conv1d(batch_size, tokens, embeddings, device):
    conv = Conv1D(embeddings, embeddings).to(device)
    qconv = QConv1D.from_module(conv)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), Calibration():
        qout = qconv(qinputs)
    # Freeze to set quantized weights
    freeze(qconv)
    # Align conv weights with quantized conv weights for comparison
    conv.weight = torch.nn.Parameter(qconv.weight.dequantize())
    conv.bias = torch.nn.Parameter(qconv.bias.dequantize())
    out = conv(qinputs.dequantize())
    q_assert_close(out, qout)
    # Now run an inference without calibrating
    with torch.no_grad():
        int_qout = qconv(qinputs)
    assert qout._scale == int_qout._scale
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1


@pytest.mark.skip("QConv1D does not work")
def test_qconv1d_serialization():
    tokens = 10
    embeddings = 32
    conv = Conv1D(embeddings, embeddings)
    qconv = QConv1D.from_module(conv)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), Calibration():
        qconv(qinputs)
    # Freeze conv to store quantized weights and biases
    qconv.freeze()
    with TemporaryDirectory() as tmpdir:
        qconv_file = os.path.join(tmpdir, "qconv.pt")
        torch.save(qconv.state_dict(), qconv_file)
        qconv_reloaded = QConv1D(embeddings, embeddings)
        # When reloading we must assign instead of copying to force quantized tensors assignment
        qconv_reloaded.load_state_dict(torch.load(qconv_file), assign=True)
    for attr in ["weight", "bias"]:
        t = getattr(qconv, attr)
        if t is not None:
            t_reloaded = getattr(qconv_reloaded, attr)
            assert torch.equal(t._data, t_reloaded._data)
            assert torch.equal(t._scale, t_reloaded._scale)
    for attr in ["in_scale", "out_scale"]:
        v = getattr(qconv, attr)
        v_reloaded = getattr(qconv_reloaded, attr)
        assert torch.equal(v, v_reloaded)


@pytest.mark.skip("QConv1D does not work")
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
def test_qconv1d_gradient(tokens, embeddings, device):
    # We use a batch size of 1 to simplify gradient manual calculations
    batch_size = 1
    conv = Conv1D(embeddings, embeddings).to(device)
    qconv = QConv1D.from_module(conv)
    assert qconv.weight.requires_grad is True
    assert qconv.bias.requires_grad is True
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    qout = qconv(qinputs)
    gradient = torch.randn(qout.size()).to(device)
    qout.backward(gradient)
    # Compute gradients manually and compare
    bias_gradient = torch.sum(gradient, axis=[0, 1])
    assert torch.allclose(qconv.bias.grad, bias_gradient)
    # FIXME: gradient calculation is wrong because of the transposed weights
    weight_gradient = torch.matmul(gradient.squeeze().t(), qinputs.dequantize().squeeze())
    assert torch.allclose(qconv.weight.grad, weight_gradient)
