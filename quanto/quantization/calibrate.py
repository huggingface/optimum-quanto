from contextlib import contextmanager

import torch
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

from .nn import QLinear
from .tensor import QuantizedTensor, scale_max


momentum = 0.9


def calibrate_input(module: torch.nn.Module, input):
    if isinstance(module, QLinear):
        # We want to requantize with the most accurate scale
        input = input[0].dequantize()
        input_scale = scale_max(input, torch.int8)
        if torch.all(module.in_scale == 1):
            module.in_scale = input_scale
        else:
            module.in_scale = momentum * module.in_scale + input_scale * (1.0 - momentum)
        return input


def calibrate_output(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    if isinstance(module, QLinear):
        output = output.dequantize()
        output = QuantizedTensor.quantize(output)
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
