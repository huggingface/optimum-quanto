import pytest
import torch

from optimum.quanto.library import extensions


extension_types = ["cpp"]
if torch.cuda.is_available():
    extension_types.append("cuda")
if torch.backends.mps.is_available():
    extension_types.append("mps")
if torch.xpu.is_available():
    extension_types.append("xpu")


@pytest.mark.parametrize("extension_type", extension_types)
def test_extension_compilation(extension_type):
    extension = getattr(extensions, extension_type).ext
    assert extension.lib is not None
