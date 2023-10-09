from contextlib import contextmanager

import torch
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

from .nn import QLinear
from .qtensor import QTensor


__all__ = ["calibration"]


momentum = 0.9


def calibrate_input(module: torch.nn.Module, input):
    if isinstance(module, QLinear):
        # We want to requantize with the most accurate scale
        input = input[0].dequantize()
        input = QTensor.quantize(input, torch.int8)
        if torch.all(module.in_scale == 1):
            module.in_scale = input._scale
        else:
            module.in_scale = momentum * module.in_scale + input._scale * (1.0 - momentum)
        return input


def calibrate_output(module: torch.nn.Module, input, output):
    if isinstance(module, QLinear):
        # Reevaluate output using float path
        input = input[0].dequantize()
        output = super(module.__class__, module).forward(input)
        # Requantize output with the most accurate scale
        output = QTensor.quantize(output, torch.int8)
        if torch.all(module.out_scale == 1):
            module.out_scale = output._scale
        else:
            module.out_scale = momentum * module.out_scale + output._scale * (1.0 - momentum)
        return output


@contextmanager
def calibration():
    """A context to calibrate quantized modules."""
    try:
        pre_handle = register_module_forward_pre_hook(calibrate_input)
        post_handle = register_module_forward_hook(calibrate_output)
        yield
    finally:
        pre_handle.remove()
        post_handle.remove()
