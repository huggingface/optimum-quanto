import pytest
import torch

from quanto import QTensor, qint8
from quanto.nn import QLinear


@pytest.mark.parametrize("in_features", [8, 16])
@pytest.mark.parametrize("out_features", [32, 64])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_qmodule_freeze(in_features, out_features, use_bias, dtype):
    qlinear = QLinear(in_features, out_features, bias=use_bias).to(dtype)
    assert not qlinear.frozen
    assert not isinstance(qlinear.weight, QTensor)
    assert qlinear.weight.dtype == dtype
    if use_bias:
        assert not isinstance(qlinear.bias, QTensor)
        assert qlinear.bias.dtype == dtype
    qweight = qlinear.qweight()
    assert isinstance(qweight, QTensor)
    assert qweight.dtype == dtype
    assert qweight.qtype == qint8
    qlinear.freeze()
    assert qlinear.frozen
    assert isinstance(qlinear.weight, QTensor)
    assert qlinear.weight.dtype == dtype
    assert qlinear.weight.qtype == qint8
    if use_bias:
        assert not isinstance(qlinear.bias, QTensor)
        assert qlinear.bias.dtype == dtype
