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

from torch import device as torch_device

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


def requantize(model, state_dict):
    # find device that model is on
    device = next(model.parameters()).device

    # empty the model params by moving to the meta device, then quantize
    model.to(torch_device("meta"))
    quantize(model)

    # move the quantized but empty model to cpu then load the state_dict
    model.to_empty(device=torch_device("cpu"))
    model.load_state_dict(state_dict)

    # move the model back to the original device
    model.to(device)


def freeze(model):
    for name, m in model.named_modules():
        if isinstance(m, QModuleMixin):
            m.freeze()
