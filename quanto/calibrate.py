from functools import partial

import torch
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._python_dispatch import TorchDispatchMode

from .nn import QModuleMixin
from .qtensor import QTensor, absmax_scale


__all__ = ["Calibration"]


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


class Calibration(TorchDispatchMode):
    """A custom torch dispatch mode to calibrate quantized modules.

    The dispatch mode tracks the outputs of each quantized module down the operations graph
    to detect which operation(s) consume(s) them.

    Baesd on that information, it decides if these outputs can be quantized or not: if
    the consuming operation only accepts higher-precision outputs, there is no point in
    quantizing them.

    If the outputs can be quantized, it evaluates if the consuming operation(s) accept
    per-axis inputs, otherwise it quantize the outputs per-tensor.

    In order to improve the accuracy of the quantized activations, the input and output
    scales of each quantized module is evaluated per-batch using the absmax algorithm and aggregated using a
    momentum.

    Args:
        momentum (`float`): the momentum to use when updating scales.
    """

    def __init__(self, *args, momentum: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        qinput = QTensor in types
        output = func(*args, **kwargs)
        if qinput and not isinstance(output, QTensor):
            # Here we will eventually modify the behaviour of the source module.
            pass
        return output

    def __enter__(self):
        super().__enter__()
        self.pre_handle = register_module_forward_pre_hook(partial(calibrate_input, momentum=self.momentum))
        self.post_handle = register_module_forward_hook(partial(calibrate_output, momentum=self.momentum))

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.pre_handle.remove()
        self.post_handle.remove()
