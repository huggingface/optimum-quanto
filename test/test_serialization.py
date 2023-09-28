import os
from tempfile import TemporaryDirectory

import pytest
import torch
from helpers import random_qtensor


def test_quantized_tensor_serialization():
    qinputs = random_qtensor((1, 10, 32), dtype=torch.float32)
    with TemporaryDirectory() as tmpdir:
        qinputs_file = os.path.join(tmpdir, "qinputs.pt")
        torch.save(qinputs, qinputs_file)
        qinputs_reloaded = torch.load(qinputs_file)
    assert torch.equal(qinputs._data, qinputs_reloaded._data)
    assert torch.equal(qinputs._scale, qinputs_reloaded._scale)


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
    with TemporaryDirectory() as tmpdir:
        qlinear_file = os.path.join(tmpdir, "qlinear.pt")
        torch.save(qlinear.state_dict(), qlinear_file)
        torch.load(qlinear_file)
        if quantize_before_load:
            qlinear_reloaded = new_qlinear(embeddings, device)
            # Since the weights are already quantized tensors, we can copy instead of assigning
            assign = False
        else:
            qlinear_reloaded = torch.nn.Linear(embeddings, embeddings)
            # We need to force assignment instead of copy to replace weights by quantized weights
            assign = True
        qlinear_reloaded.load_state_dict(torch.load(qlinear_file), assign=assign)
    for attr in ["weight", "bias"]:
        t = getattr(qlinear, attr)
        t_reloaded = getattr(qlinear_reloaded, attr)
        assert torch.equal(t._data, t_reloaded._data)
        assert torch.equal(t._scale, t_reloaded._scale)
