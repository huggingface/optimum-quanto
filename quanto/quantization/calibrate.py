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


def update_scale(scale, new_scale, momentum):
    if torch.all(scale == 1):
        # The original scale has never been set
        return new_scale
    return momentum * scale + new_scale * (1.0 - momentum)


def calibrate_input(module: torch.nn.Module, input, momentum: float = 0.9):
    if isinstance(module, QModuleMixin):
        input = input[0]
        if isinstance(input, QTensor):
            # Just adopt the maximum scale of the input
            module.in_scale = torch.max(input._scale)
        else:
            # Evaluate the best scale
            input_scale = absmax_scale(input, torch.int8)
            # Update the module input scale accordingly
            module.in_scale = update_scale(module.in_scale, input_scale, momentum)
        return input


def calibrate_output(
    module: torch.nn.Module,
    input,
    output,
    momentum: float = 0.9,
    per_axis: bool = False,
):
    if isinstance(module, (QModuleMixin)):
        # Reevaluate output using float path and get its actual scale
        float_input = input[0]
        if isinstance(float_input, QTensor):
            float_input = float_input.dequantize()
        float_output = super(module.__class__, module).forward(float_input)
        output_scale = absmax_scale(float_output, torch.int8, axis=-1 if per_axis else None)
        # Update the module output scale accordingly
        module.out_scale = update_scale(module.out_scale, output_scale, momentum)
        # Reevaluate output with the correct output scale
        qoutput = module.forward(input[0])
        return qoutput


@contextmanager
def calibration(momentum: float = 0.9, per_axis: bool = False):
    """A context to evaluate the quantized modules input and output scales.

    Scales are evaluated per-batch using the absmax algorithm and aggregated using a
    momentum.
    Input scales are always evaluated per-tensor, but output scales can be evaluated
    along the last dimension of the output.

    Args:
        momentum (`float`): the momentum to use when updating scales.
        per_axis (`bool`): evaluate output scales along the last dimension of the outputs. Defaults to False.
    """
    try:
        pre_handle = register_module_forward_pre_hook(partial(calibrate_input, momentum=momentum))
        post_handle = register_module_forward_hook(partial(calibrate_output, momentum=momentum, per_axis=per_axis))
        yield
    finally:
        pre_handle.remove()
        post_handle.remove()
