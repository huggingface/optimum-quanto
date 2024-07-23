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

from optimum.quanto import (
    qint8,
    quantize,
)
from optimum.quanto.nn import QModuleMixin


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.mid_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = torch.nn.functional.relu(self.input_layer(inputs))
        x = torch.nn.functional.relu(self.mid_layer(x))
        return self.output_layer(x)


class ClassificationModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, classes):
        super().__init__()
        self.model = MLP(input_size, output_size, hidden_size)
        self.lm_head = torch.nn.Linear(output_size, classes)

    def forward(self, inputs):
        x = self.model(inputs)
        return torch.nn.functional.softmax(self.classifier(x), dim=-1)


def has_children(module: torch.nn.Module):
    return next(module.children(), None) is not None


def leaf_module_names(module: torch.nn.Module):
    return [name for name, m in module.named_modules() if not has_children(m)]


def parent_module_names(module: torch.nn.Module):
    return [name for name, m in module.named_children() if has_children(m)]


def test_quantize_mlp_include_explicit_layers():
    model = ClassificationModel(32, 10, 128, 10)
    include_names = leaf_module_names(model)
    for include in include_names:
        model = ClassificationModel(32, 10, 128, 10)
        quantize(model, weights=qint8, include=include)
        for name, m in model.named_modules():
            if name == include:
                assert isinstance(m, QModuleMixin)
            else:
                assert not isinstance(m, QModuleMixin)


def test_quantize_mlp_exclude_explicit_layers():
    model = ClassificationModel(32, 10, 128, 10)
    exclude_names = leaf_module_names(model)
    for exclude in exclude_names:
        model = ClassificationModel(32, 10, 128, 10)
        quantize(model, weights=qint8, exclude=exclude)
        for name, m in model.named_modules():
            if name == exclude:
                assert not isinstance(m, QModuleMixin)
            elif not has_children(m):
                assert isinstance(m, QModuleMixin)


def test_quantize_mlp_include_layer_patterns():
    model = ClassificationModel(32, 10, 128, 10)
    parent_names = parent_module_names(model)
    for parent_name in parent_names:
        model = ClassificationModel(32, 10, 128, 10)
        quantize(model, weights=qint8, include=f"{parent_name}*")
        for name, m in model.named_modules():
            if name.startswith(parent_name) and not has_children(m):
                assert isinstance(m, QModuleMixin)
            else:
                assert not isinstance(m, QModuleMixin)


def test_quantize_mlp_exclude_layer_patterns():
    model = ClassificationModel(32, 10, 128, 10)
    parent_names = parent_module_names(model)
    for parent_name in parent_names:
        model = ClassificationModel(32, 10, 128, 10)
        quantize(model, weights=qint8, exclude=f"{parent_name}*")
        for name, m in model.named_modules():
            if name.startswith(parent_name):
                assert not isinstance(m, QModuleMixin)
            elif not has_children(m):
                assert isinstance(m, QModuleMixin)
