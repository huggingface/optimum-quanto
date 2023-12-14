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


def pytest_configure(config):
    # register additional markers
    config.addinivalue_line("markers", "skip_device(type): mark test to be skipped for the specified device type")


def pytest_runtest_call(item):
    fixture_name = "device"
    if fixture_name in item.fixturenames:
        # TODO: should be able to recover the fixture id instead of the actual value
        fixture_arg = item.funcargs[fixture_name].type
        skip_marks = {mark.args[0] for mark in item.iter_markers(name=f"skip_{fixture_name}")}
        if fixture_arg in skip_marks:
            pytest.skip(f"Test skipped for {fixture_name} {fixture_arg}")
