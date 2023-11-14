from contextlib import contextmanager
from functools import partial
from typing import Callable, Optional, Union

import torch
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

from .nn import QModuleMixin
from .qtensor import QTensor, absmax_scale


__all__ = ["calibration"]


# Calibration callback with the following arguments:
# - quantized module,
# - inputs (can be quantized or not)
# - outputs resulting from a float inference,
# - quantized outputs from a quantized inference.
CalibrationHook = Callable[[QModuleMixin, Union[torch.Tensor, QTensor], torch.Tensor, QTensor], None]


def update_scale(scale, new_scale, momentum):
    if torch.all(scale == 1):
        # The original scale has never been set
        return new_scale
    return momentum * scale + new_scale * (1.0 - momentum)


def calibrate_input(module: torch.nn.Module, input, momentum: float = 0.9):
    if isinstance(module, QModuleMixin):
        # Evaluate the actual scale of the input
        input = input[0]
        input_qtensor = isinstance(input, QTensor)
        if input_qtensor:
            input = input.dequantize()
        input_scale = absmax_scale(input, torch.int8)
        # Update the module input scale accordingly
        module.in_scale = update_scale(module.in_scale, input_scale, momentum)
        if input_qtensor:
            # Requantize input with the updated input scale
            return QTensor.quantize(input, torch.int8, module.in_scale)
        return input


def calibrate_output(
    module: torch.nn.Module,
    input,
    output,
    momentum: float = 0.9,
    hook: Optional[CalibrationHook] = None,
):
    if isinstance(module, (QModuleMixin)):
        # Reevaluate output using float path and get its actual scale
        float_input = input[0]
        if isinstance(float_input, QTensor):
            float_input = float_input.dequantize()
        float_output = super(module.__class__, module).forward(float_input)
        output_scale = absmax_scale(float_output, torch.int8)
        # Update the module output scale accordingly
        module.out_scale = update_scale(module.out_scale, output_scale, momentum)
        # Reevaluate output with the correct output scale
        qoutput = module.forward(input[0])
        if hook is not None:
            hook(module, input[0], float_output, qoutput)
        return qoutput


@contextmanager
def calibration(momentum: float = 0.9, hook: Optional[CalibrationHook] = None):
    """A context to calibrate quantized modules."""
    try:
        pre_handle = register_module_forward_pre_hook(partial(calibrate_input, momentum=momentum))
        post_handle = register_module_forward_hook(partial(calibrate_output, momentum=momentum, hook=hook))
        yield
    finally:
        pre_handle.remove()
        post_handle.remove()
