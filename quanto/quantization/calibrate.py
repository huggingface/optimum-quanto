from contextlib import contextmanager
from functools import partial

import torch
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

from .nn import QModuleMixin
from .qtensor import QTensor, absmax_scale


__all__ = ["calibration"]


def calibrate_input(module: torch.nn.Module, input, momentum: float = 0.9):
    if isinstance(module, QModuleMixin):
        # Evaluate the actual scale of the input
        input = input[0]
        input_qtensor = isinstance(input, QTensor)
        if input_qtensor:
            input = input.dequantize()
        input_scale = absmax_scale(input, torch.int8)
        # Update the module input scale accordingly
        if torch.all(module.in_scale == 1):
            module.in_scale = input_scale
        else:
            module.in_scale = momentum * module.in_scale + input_scale * (1.0 - momentum)
        if input_qtensor:
            # Requantize input with the updated input scale
            return QTensor.quantize(input, torch.int8, module.in_scale)
        return input


def calibrate_output(module: torch.nn.Module, input, output, momentum=0.9):
    if isinstance(module, (QModuleMixin)):
        print(momentum)
        # Reevaluate output using float path and get its actual scale
        input = input[0]
        if isinstance(input, QTensor):
            input = input.dequantize()
        output = super(module.__class__, module).forward(input)
        output_scale = absmax_scale(output, torch.int8)
        # Update the module output scale accordingly
        if torch.all(module.out_scale == 1):
            module.out_scale = output_scale
        else:
            module.out_scale = momentum * module.out_scale + output_scale * (1.0 - momentum)
        # Requantize output with the updated output scale
        return QTensor.quantize(output, torch.int8, module.out_scale)


@contextmanager
def calibration(momentum=0.9):
    """A context to calibrate quantized modules."""
    try:
        pre_handle = register_module_forward_pre_hook(partial(calibrate_input, momentum=momentum))
        post_handle = register_module_forward_hook(partial(calibrate_output, momentum=momentum))
        yield
    finally:
        pre_handle.remove()
        post_handle.remove()
