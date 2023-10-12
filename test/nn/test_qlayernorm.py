import os
from tempfile import TemporaryDirectory

import pytest
import torch
from helpers import q_assert_close, random_qtensor

from quanto.quantization import calibration
from quanto.quantization.nn import QLayerNorm


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(32, 32), (10, 32)])
def test_quantize_layernorm(batch_size, tokens, embeddings, device):
    # Instantiate a normalization layer
    norm = torch.nn.LayerNorm(embeddings).to(device)
    qnorm = QLayerNorm.from_module(norm)
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), calibration():
        qout = qnorm(qinputs)
    out = norm(qinputs.dequantize())
    q_assert_close(out, qout)
    # Now run an inference without calibrating
    with torch.no_grad():
        int_qout = qnorm(qinputs)
    assert qout._scale == int_qout._scale
    # There may be a slight difference, but of at most one quantization interval
    assert torch.max(torch.abs(qout._data - int_qout._data)) <= 1


def test_qnorm_serialization():
    tokens = 10
    embeddings = 32
    norm = torch.nn.LayerNorm(embeddings)
    qnorm = QLayerNorm.from_module(norm)
    qinputs = random_qtensor((1,) + (tokens, embeddings), dtype=torch.float32)
    # Calibrate and obtain quantized outputs
    with torch.no_grad(), calibration():
        qnorm(qinputs)
    with TemporaryDirectory() as tmpdir:
        qnorm_file = os.path.join(tmpdir, "qnorm.pt")
        torch.save(qnorm.state_dict(), qnorm_file)
        qnorm_reloaded = QLayerNorm(embeddings)
        # When reloading we must assign instead of copying to force quantized tensors assignment
        qnorm_reloaded.load_state_dict(torch.load(qnorm_file), assign=True)
    for attr in ["in_scale", "out_scale"]:
        v = getattr(qnorm, attr)
        v_reloaded = getattr(qnorm_reloaded, attr)
        assert torch.equal(v, v_reloaded)
