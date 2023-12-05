import pytest
import torch
from helpers import q_assert_close, random_qtensor, random_tensor

from quanto.quantization import QTensor


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("weight_axis", [None, 0], ids=["per-tensor", "per-axis"])
def test_linear(batch_size, tokens, embeddings, use_bias, weight_axis, device):
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=torch.float32).to(device)
    qweight = random_qtensor((embeddings, embeddings), dtype=torch.float32, axis=weight_axis).to(device)
    if use_bias:
        bias = random_tensor((embeddings,), dtype=torch.float32).to(device)
        # Bias must be quantized to int32 with the same scale as the product of the two int8
        prod_scale = torch.squeeze(qinputs._scale * qweight._scale)
        qbias = QTensor.quantize(bias, torch.int32, prod_scale)
    else:
        qbias = None
    out = torch.nn.functional.linear(
        qinputs.dequantize(), qweight.dequantize(), qbias.dequantize() if use_bias else None
    )
    qout = torch.nn.functional.linear(qinputs, qweight, qbias)
    q_assert_close(out, qout)
