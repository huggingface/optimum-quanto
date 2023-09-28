import pytest
import torch


devices = ["cpu"]
if torch.cuda.is_available():
    devices += ["cuda"]


@pytest.fixture(scope="module", params=devices)
def device(request):
    return torch.device(request.param)
