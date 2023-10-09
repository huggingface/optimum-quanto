import pytest
import torch


devices = ["cpu"]
if torch.cuda.is_available():
    devices += ["cuda"]
elif torch.backends.mps.is_available():
    devices += ["mps"]


@pytest.fixture(scope="module", params=devices)
def device(request):
    return torch.device(request.param)
