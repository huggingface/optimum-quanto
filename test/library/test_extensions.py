import platform

import pytest
import torch
from packaging import version

from optimum.quanto.library.extensions import get_extension, is_extension_available


def _is_xpu_available():
    # SYCL extension support is added in torch>=2.7 on Linux
    if platform.system() != "Linux":
        return False
    if version.parse(torch.__version__).release < version.parse("2.7").release:
        return False
    return torch.xpu.is_available()


extension_names = ["quanto_cpp"]
if torch.cuda.is_available():
    if torch.version.cuda:
        extension_names.append("quanto_cuda")
    if torch.version.hip:
        extension_names.append("quanto_hip")
if torch.backends.mps.is_available():
    extension_names.append("quanto_mps")
if _is_xpu_available():
    extension_names.append("quanto_xpu")


@pytest.mark.parametrize("extension_name", extension_names)
def test_extension_available(extension_name):
    assert is_extension_available(extension_name)


@pytest.mark.parametrize("extension_name", extension_names)
def test_extension_compilation(extension_name):
    extension = get_extension(extension_name)
    assert extension.lib is not None
