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
from helpers import random_tensor

from optimum.quanto import AbsmaxOptimizer, qint8, quantize_weight


def test_weight_qbytes_tensor_requires_grad(device):
    w = random_tensor((10, 10), dtype=torch.float32).to(device)
    w.requires_grad = True
    scale = AbsmaxOptimizer()(w, qtype=qint8, axis=0)
    qw = quantize_weight(w, qtype=qint8, axis=0, scale=scale)
    assert qw.requires_grad is True


def test_weight_qbytes_tensor_backward(device):
    w = random_tensor((10, 10), dtype=torch.float32).to(device)
    w.requires_grad = True
    scale = AbsmaxOptimizer()(w, qtype=qint8, axis=0)
    qw = quantize_weight(w, qtype=qint8, axis=0, scale=scale)
    gradient = torch.randn((10, 10)).to(device)
    # Backpropagate gradient to the inner float weights
    qw.dequantize().backward(gradient)
    assert torch.equal(w.grad, gradient)


def test_weight_qbytes_tensor_chained_backward(device):
    a = random_tensor((10, 10), dtype=torch.float32).to(device)
    a.requires_grad = True
    scale = AbsmaxOptimizer()(a, qtype=qint8, axis=0)
    qa = quantize_weight(a, qtype=qint8, axis=0, scale=scale)
    b = random_tensor((10, 10), dtype=torch.float32).to(device)
    b.requires_grad = True
    scale = AbsmaxOptimizer()(b, qtype=qint8, axis=0)
    qb = quantize_weight(b, qtype=qint8, axis=0, scale=scale)
    # Evaluate the product
    prod = qa * qb
    # Backpropagate
    gradient = torch.randn((10, 10)).to(device)
    prod.backward(gradient)
    assert torch.allclose(a.grad, qb.dequantize() * gradient)
    assert torch.allclose(b.grad, qa.dequantize() * gradient)
