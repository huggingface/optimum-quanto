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

import torch

from .nn import QModuleMixin, quantize_module


__all__ = ["quantize", "freeze", "requantize"]


def set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)


def quantize(model, modules=None, **kwargs):
    # Quantization happens in-place
    for name, m in model.named_modules():
        if modules is not None and m not in modules:
            continue
        qmodule = quantize_module(m, **kwargs)
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
