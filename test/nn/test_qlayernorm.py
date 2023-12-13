import pytest
import torch
from helpers import q_assert_close, random_qtensor

from quanto.quantization import QTensor, calibration
from quanto.quantization.nn import QLayerNorm


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("activations", [torch.int8], ids=["a-int8"])
def test_quantize_layernorm(batch_size, tokens, embeddings, dtype, activations, device):
    if dtype == torch.float16 and device.type == "cpu":
        pytest.skip("layer_norm is not supported for float16 on CPU")
    # Instantiate a normalization layer
    norm = torch.nn.LayerNorm(embeddings).to(dtype).to(device)
    qnorm = QLayerNorm.from_module(norm, activations=activations)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), itype=activations, dtype=dtype).to(device)
    # Calibrate to avoid clipping and to set the correct dtype
    with torch.no_grad(), calibration():
        qout = qnorm(qinputs)
    qout = qnorm(qinputs)
    assert isinstance(qout, QTensor)
    assert qout.dtype == dtype
    assert qout.itype == activations
    # Compare with the float results
    out = norm(qinputs.dequantize())
    q_assert_close(out, qout)


def test_quantize_layernom_no_activation():
    norm = torch.nn.LayerNorm(32)
    qnorm = QLayerNorm.from_module(norm, activations=None)
    assert qnorm is None
