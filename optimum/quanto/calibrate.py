# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.overrides import TorchFunctionMode

from .nn import QModuleMixin
from .tensor import ActivationQBytesTensor, QTensor, axis_to_dim, dtype_info, qint8, qtype


__all__ = ["Calibration", "absmax_scale"]


def _updated_scale(scale, new_scale, momentum):
    if torch.all(scale == 1):
        return new_scale
    return momentum * scale + new_scale * (1.0 - momentum)


def absmax_scale(base: torch.Tensor, qtype: qtype = qint8, axis: Optional[int] = None) -> torch.Tensor:
    """Evaluate the quantization scale using the absmax algorithm.

    The Absolute Maximum quantization algorithm is a symmetrical quantization
    algorithm where the scale corresponds to the maximum absolute value of the
    base divided by the highest positive integer value for the target integer
    representation.

    Args:
        base (`torch.Tensor`): the base tensor on which the scale will be applied.
        qtype (`quanto.qtype`): the target qtype for quantization.
        axis (`int`): the index of the axis to preserve, or -1 for the last one.
            Defaults to None to reduce all axis.

    Returns:
        `torch.Tensor`: a scale tensor of the same dtype as the base.
    """
    base = torch.abs(base)
    if axis is None:
        qranges = torch.max(base)
    else:
        dim = axis_to_dim(base, axis)
        qranges = torch.amax(base, dim=dim, keepdim=True)
    info = dtype_info(qtype.dtype)
    return qranges / info.max


class Calibration(TorchFunctionMode):
    """A custom torch dispatch mode to calibrate quantized modules.

    In order to improve the accuracy of the quantized activations, the input and output
    scales of each quantized module are evaluated per-batch using the absmax algorithm and aggregated using a
    momentum.

    The dispatch mode also tracks the calls to each torch function down the model graph, and applies optional
    optimizations:
    - streamline: do not quantize activations that are immediately consumed by an incompatible function (like `add` or `silu`).

    Args:
        momentum (`float`): the momentum to use when updating scales.
        streamline (`bool`): if True, avoid quantizing activations when they are consumed by an incompatible function. Defaults to True.
        debug (`bool`): provide very verbose feedback on the console during calibration.
    """

    def __init__(self, *args, momentum: float = 0.9, streamline=True, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.streamline = streamline
        if streamline:
            self.modules_qactivations = {}
            self.streamline_hooks = {}
        self.debug = debug

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        qinput = QTensor in types
        output = func(*args, **kwargs)
        if self.streamline and qinput:
            for i, arg in enumerate(args):
                module = getattr(arg, "src_module", None)
                if module is not None:
                    if isinstance(output, ActivationQBytesTensor):
                        # Quantized activations are required for that module
                        self.modules_qactivations[module] = True
                    elif isinstance(output, torch.Tensor):
                        # Quantized activations are not required for that module unless another function requires them
                        qactivations_required = self.modules_qactivations.get(module, False)
                        self.modules_qactivations[module] = qactivations_required
        return output

    def __enter__(self):
        super().__enter__()
        self.pre_handle = register_module_forward_pre_hook(self.calibrate_input)
        self.post_handle = register_module_forward_hook(self.calibrate_output)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.pre_handle.remove()
        self.post_handle.remove()
        if self.streamline:
            for handle in self.streamline_hooks.values():
                handle.remove()

    def calibrate_input(self, module: torch.nn.Module, input, momentum: float = 0.9):
        """Calibrate a module input scale

        This is registered as a global hook that is called before any module forward pre hook.
        """
        if isinstance(module, QModuleMixin) and module.activation_qtype is not None:
            input = input[0]
            if isinstance(input, ActivationQBytesTensor):
                # Just adopt the maximum scale of the input
                module.input_scale = torch.max(input._scale)
            else:
                # Evaluate the best scale
                input_scale = absmax_scale(input, module.activation_qtype)
                module.input_scale = _updated_scale(module.input_scale, input_scale, momentum)
            if self.streamline and module not in self.streamline_hooks:
                # Add a hook to tag the module outputs (after the module quantization hook in QModuleMixin)
                self.streamline_hooks[module] = module.register_forward_hook(self.tag_outputs)
            return input

    def calibrate_output(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        """Calibrate a module output scale

        This is registered as a global hook that is called before any module forward hook.

        When the module is a QModuleMixin, its outputs are not quantized yet because they
        are only quantized in the QModuleMixin.quantize_output forward hook.
        """
        if isinstance(module, (QModuleMixin)) and module.activation_qtype is not None:
            # Evaluate the optimal scale per-tensor and update output scale
            output_scale = absmax_scale(output, module.activation_qtype, axis=None)
            module.output_scale = _updated_scale(module.output_scale, output_scale, self.momentum)
            return output
        else:
            if self.streamline:
                for name, child in module.named_children():
                    if isinstance(child, QModuleMixin) and child.activation_qtype is not None:
                        qactivations_required = self.modules_qactivations.get(child, False)
                        if not qactivations_required:
                            # Disable output quantization for this child as its outputs are only consumed by incompatible functions.
                            child.disable_output_quantization()
            if self.debug:
                for name, child in module.named_children():
                    if isinstance(child, QModuleMixin):
                        classname = child.__class__.__name__
                        trace = f"{name}({classname}) activations are"
                        if child.activation_qtype is None:
                            trace += " not quantized."
                        else:
                            trace += f" quantized to {child.activation_qtype} with scale {child.output_scale}."
                        print(trace)

    def tag_outputs(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        """Mark outputs as generated by a module

        This is called as a module forward hook that is called after the QModuleMixin.quantize_output
        forward hook.

        This is useful in streamline mode to identify the module that generated a specific QTensor.
        """
        output.src_module = module
