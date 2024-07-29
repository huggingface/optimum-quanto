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

from ..qtype import qtype
from .qbytes import ActivationQBytesTensor


__all__ = ["quantize_activation"]


def quantize_activation(t: torch.Tensor, qtype: qtype, scale: torch.Tensor):
    """Quantize an activation Tensor.

    Activations are always quantized per-tensor with a scalar scale.

    Args:
        base (`torch.Tensor`): the Tensor to quantize
        qtype (`quanto.qtype`): The target quantization type
        scale (`torch.Tensor`): The scalar quantization scale

    Returns:
        A quantized Tensor.
    """
    if scale.numel() != 1:
        raise ValueError("Parameter scale must be a scalar because activations can only be quantized per-tensor")
    return ActivationQBytesTensor.quantize(t, qtype, scale)
