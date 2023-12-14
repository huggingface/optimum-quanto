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


def updated_scale(scale, new_scale, momentum):
    if torch.all(scale == 1):
        return new_scale
    return momentum * scale + new_scale * (1.0 - momentum)


def calibrate_input(module: torch.nn.Module, input, momentum: float = 0.9):
    if isinstance(module, QModuleMixin) and module.activations is not None:
        input = input[0]
        if isinstance(input, QTensor):
            # Just adopt the maximum scale of the input
            module.input_scale = torch.max(input._scale)
        else:
            # Evaluate the best scale
            input_scale = absmax_scale(input, module.activations)
            module.input_scale = updated_scale(module.input_scale, input_scale, momentum)
        return input


def calibrate_output(
    module: torch.nn.Module,
    input,
    output,
    momentum: float = 0.9,
):
    if isinstance(module, (QModuleMixin)) and module.activations is not None:
        # Reevaluate raw module output
        qoutput = module.qforward(input[0])
        if isinstance(qoutput, QTensor):
            qoutput = qoutput.dequantize()
        # Evaluate the optimal scale per-tensor and update output scale
        output_scale = absmax_scale(qoutput, module.activations, axis=None)
        module.output_scale = updated_scale(module.output_scale, output_scale, momentum)
        # Reevaluate output with the correct output scale
        return module.forward(input[0])


@contextmanager
def calibration(momentum: float = 0.9):
    """A context to evaluate the quantized modules input and output scales.

    Scales are evaluated per-batch using the absmax algorithm and aggregated using a
    momentum.
    Input and output scales are always evaluated per-tensor.

    Args:
        momentum (`float`): the momentum to use when updating scales.
    """
    try:
        pre_handle = register_module_forward_pre_hook(partial(calibrate_input, momentum=momentum))
        post_handle = register_module_forward_hook(partial(calibrate_output, momentum=momentum))
        yield
    finally:
        pre_handle.remove()
        post_handle.remove()
