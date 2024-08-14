import pytest
import torch

from optimum.quanto.library.extensions import get_extension, is_extension_available


extension_names = ["quanto_cpp"]
if torch.cuda.is_available():
    if torch.version.cuda:
        extension_names.append("quanto_cuda")
    if torch.version.hip:
        extension_names.append("quanto_hip")
if torch.backends.mps.is_available():
    extension_names.append("quanto_mps")


@pytest.mark.parametrize("extension_name", extension_names)
def test_extension_available(extension_name):
    assert is_extension_available(extension_name)


@pytest.mark.parametrize("extension_name", extension_names)
def test_extension_compilation(extension_name):
    extension = get_extension(extension_name)
    assert extension.lib is not None
