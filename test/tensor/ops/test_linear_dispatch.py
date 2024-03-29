import pytest
import torch
from helpers import assert_similar, random_qactivation, random_qweight, random_tensor

from quanto import qint2, qint4, qint8


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("weight_qtype", [qint2, qint4, qint8], ids=["qint2", "qint4", "qint8"])
def test_qactivation_qweight_linear(batch_size, tokens, embeddings, use_bias, dtype, weight_qtype, device):
    if dtype == torch.float16 and device == torch.device("cpu"):
        pytest.skip("torch.ops.aten.addmm is not supported for float16 on CPU.")
    qinputs = random_qactivation((batch_size,) + (tokens, embeddings), dtype=dtype).to(device)
    qweight = random_qweight((embeddings, embeddings), weight_qtype, dtype=dtype, axis=0).to(device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    out = torch.nn.functional.linear(qinputs.dequantize(), qweight.dequantize(), bias)
    qout = torch.nn.functional.linear(qinputs, qweight, bias)
    assert_similar(out, qout)
