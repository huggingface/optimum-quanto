import torch

from .cpp import *


if torch.backends.mps.is_available():
    from .mps import *
