import pytest
import torch
from helpers import random_qweight

from quanto import qint8


@pytest.mark.parametrize("axis", [0, -1], ids=["first-axis", "last-axis"])
@pytest.mark.parametrize(
    "group_size",
    [None, 2],
    ids=["channel-wise", "group-wise"],
)
def test_qweight_transpose_2d(axis, group_size, device):
    input_shape = (4, 6)
    qinputs = random_qweight(input_shape, qint8, axis=axis, group_size=group_size).to(device)
    qtransposed = qinputs.t()
    assert qtransposed.qtype == qinputs.qtype
    if axis == -1:
        assert qtransposed.axis == 0
    elif axis == 0:
        assert qtransposed.axis == -1
    assert qtransposed.shape == input_shape[::-1]
    assert torch.equal(qtransposed.dequantize(), qinputs.dequantize().t())
