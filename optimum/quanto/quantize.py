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

from fnmatch import fnmatch
from typing import List, Optional, Union

import torch

from .nn import QModuleMixin, quantize_module
from .tensor import Optimizer, qtype


__all__ = ["quantize", "freeze", "requantize"]


def set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)


def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else exclude
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(fnmatch(name, pattern) for pattern in include):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        qmodule = quantize_module(m, weights=weights, activations=activations, optimizer=optimizer)
        if qmodule is not None:
            set_module_by_name(model, name, qmodule)
            qmodule.name = name
            for name, param in m.named_parameters():
                # Save device memory by clearing parameters
                setattr(m, name, None)
                del param


def requantize(model, state_dict, device=None):
    # Evaluate the model current device
    current_device = next(model.parameters()).device
    if current_device.type != "meta":
        # empty the model params by moving to the meta device
        model.to(torch.device("meta"))
        if device is None:
            device = current_device

    # Quantize the model without parameters to create blank quantized modules
    quantize(model)

    # Move the quantized but empty model to the CPU device to avoid creating large weights on the device
    model.to_empty(device=torch.device("cpu"))
    #  Load the state_dict, applying quantization parameters and thus reducing model weights
    model.load_state_dict(state_dict)

    if device is not None:
        model.to(device)


def freeze(model):
    for name, m in model.named_modules():
        if isinstance(m, QModuleMixin):
            m.freeze()
