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
from typing import Any, Dict, List, Optional, Union

import torch

from .nn import QModuleMixin, quantize_module
from .tensor import Optimizer, qtype


__all__ = ["quantize", "freeze", "requantize", "quantization_map"]


def set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)


def _quantize_submodule(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
    weights: Optional[Union[str, qtype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
):
    qmodule = quantize_module(module, weights=weights, activations=activations, optimizer=optimizer)
    if qmodule is not None:
        set_module_by_name(model, name, qmodule)
        qmodule.name = name
        for name, param in module.named_parameters():
            # Save device memory by clearing parameters
            setattr(module, name, None)
            del param


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
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(fnmatch(name, pattern) for pattern in include):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        _quantize_submodule(model, name, m, weights=weights, activations=activations, optimizer=optimizer)


def requantize(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    quantization_map: Dict[str, Dict[str, str]],
    device: torch.device = None,
):
    if device is None:
        device = next(model.parameters()).device
        if device.type == "meta":
            device = torch.device("cpu")

    # Quantize the model with parameters from the quantization map
    for name, m in model.named_modules():
        qconfig = quantization_map.get(name, None)
        if qconfig is not None:
            weights = qconfig["weights"]
            if weights == "none":
                weights = None
            activations = qconfig["activations"]
            if activations == "none":
                activations = None
            _quantize_submodule(model, name, m, weights=weights, activations=activations)

    # Move model parameters and buffers to CPU before materializing quantized weights
    for name, m in model.named_modules():

        def move_tensor(t, device):
            if t.device.type == "meta":
                return torch.empty_like(t, device=device)
            return t.to(device)

        for name, param in m.named_parameters(recurse=False):
            setattr(m, name, torch.nn.Parameter(move_tensor(param, "cpu")))
        for name, param in m.named_buffers(recurse=False):
            setattr(m, name, move_tensor(param, "cpu"))

    # Move to target device
    model.to(device)
    # Load the quantized model weights
    model.load_state_dict(state_dict, strict=False)


def freeze(model):
    for name, m in model.named_modules():
        if isinstance(m, QModuleMixin):
            m.freeze()


def quantization_map(model: torch.nn.Module) -> Dict[str, Dict[str, str]]:
    """Returns the quantization map of a module

    The quantization map is a dictionary of quantization parameters indexed
    by the module submodule names (including prefix).

    This is mainly used for serialization.

    Args:
        model (`torch.nn.Module`): the root module to map.

    Returns:
        a dictionary of quantization parameters indexed by layer names.
    """
    config = {}
    for name, m in model.named_modules():
        if isinstance(m, QModuleMixin):
            config[name] = {
                "weights": "none" if m.weight_qtype is None else m.weight_qtype.name,
                "activations": "none" if m.activation_qtype is None else m.activation_qtype.name,
            }
    return config
