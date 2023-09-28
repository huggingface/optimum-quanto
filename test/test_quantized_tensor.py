import os
from tempfile import TemporaryDirectory

import pytest
import torch
from helpers import q_assert_close, random_qtensor, random_tensor

from quanto.quantization.tensor import QuantizedTensor


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("int_dtype", [torch.int8, torch.int16], ids=["int8", "int16"])
def test_quantize_dequantize(input_shape, int_dtype, device):
    a = random_tensor(input_shape, dtype=torch.float32).to(device)
    qa = QuantizedTensor.quantize(a, int_dtype)
    q_assert_close(a, qa)


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
@pytest.mark.parametrize("int_dtype", [torch.int8, torch.int16, torch.int32], ids=["int8", "int16", "int32"])
def test_instantiate(input_shape, int_dtype, device):
    max_value = min(1024, torch.iinfo(int_dtype).max)
    data = torch.randint(-max_value, max_value, input_shape, dtype=int_dtype)
    qa = QuantizedTensor(data, scale=torch.tensor(1.0 / max_value)).to(device)
    assert torch.max(torch.abs(qa.dequantize())) <= 1


@pytest.mark.parametrize("input_shape", [(10,), (1, 10), (10, 32, 32)])
def test_rescale(input_shape, device):
    int_max_value = 1000
    data = torch.randint(-int_max_value, int_max_value, input_shape, dtype=torch.int32)
    scale = torch.tensor(1.0 / int_max_value)
    qa = QuantizedTensor(data, scale).to(device)
    # Get the actual maximum
    a = qa.dequantize()
    float_max_value = torch.max(torch.abs(a))
    assert float_max_value <= 1
    # Rescale to int8
    qa_rescaled = qa.rescale(float_max_value / torch.iinfo(torch.int8).max, torch.int8)
    q_assert_close(a, qa_rescaled)


def test_quantized_tensor_serialization():
    qinputs = random_qtensor((1, 10, 32), dtype=torch.float32)
    with TemporaryDirectory() as tmpdir:
        qinputs_file = os.path.join(tmpdir, "qinputs.pt")
        torch.save(qinputs, qinputs_file)
        qinputs_reloaded = torch.load(qinputs_file)
    assert torch.equal(qinputs._data, qinputs_reloaded._data)
    assert torch.equal(qinputs._scale, qinputs_reloaded._scale)
