import pytest
import torch
from helpers import q_assert_close, random_qtensor, random_tensor

from quanto import QTensor, qint16


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("weight_axis", [None, 0], ids=["per-tensor", "per-axis"])
def test_linear(batch_size, tokens, embeddings, use_bias, dtype, weight_axis, device):
    if dtype == torch.float16 and device == torch.device("cpu"):
        pytest.skip("torch.ops.aten.addmm is not supported for float16 on CPU.")
    qinputs = random_qtensor((batch_size,) + (tokens, embeddings), dtype=dtype).to(device)
    qweight = random_qtensor((embeddings, embeddings), dtype=dtype, axis=weight_axis).to(device)
    if use_bias:
        bias = random_tensor((embeddings,), dtype=dtype).to(device)
        # Bias must be quantized to int16 with the same scale as the product of the two int8
        prod_scale = torch.squeeze(qinputs._scale * qweight._scale)
        qbias = QTensor.quantize(bias, qint16, prod_scale)
    else:
        qbias = None
    out = torch.nn.functional.linear(
        qinputs.dequantize(), qweight.dequantize(), qbias.dequantize() if use_bias else None
    )
    qout = torch.nn.functional.linear(qinputs, qweight, qbias)
    assert isinstance(qout, QTensor)
    # We need to increase rtol for float16
    rtol = {torch.float32: 1e-5, torch.float16: 1e-2}[dtype]
    q_assert_close(out, qout, rtol)
