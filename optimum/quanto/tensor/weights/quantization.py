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

from ..qtype import qtype
from .qbits import WeightQBitsTensor
from .qbytes import WeightQBytesTensor


__all__ = ["quantize_weight"]


def quantize_weight(
    t: torch.Tensor,
    qtype: qtype,
    axis: int,
    scale: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    activation_qtype: Optional[qtype] = None,
    optimized: Optional[bool] = True,
):
    """Quantize a weight Tensor.

    Weights are always quantized per-axis.

    Args:
        t (`torch.Tensor`): the weight Tensor to quantize
        qtype (`quanto.qtype`): The target quantization type
        axis ('int`): The quantization axis (0 or -1)
        scale (`torch.Tensor`): the quantization scale
        shift (`Optional[torch.Tensor]`): optional shift to apply
        group_size (`Optional[int]`): The quantization group size
        activation_qtype (`Optional[qtype]`, defaults to `None`):
            Which quantization type is being used for the activations. The function `quantize_weight`
            initializes `torch.Tensor` subclasses that may depend on the activation dtype.
            `None` corresponds to no quantization.
        optimized (`Optional[bool]`, defaults to True):
            If True, the quantization algorithm will select the most efficient kernel
            for the weights and format the resulting Tensor accordingly.
            If False, a kernel-agnostic Tensor will be returned (but it can be optimized later
            explicitly by calling QTensor.optimize() or implicitly by moving it to a specific device).
    Returns:
        A quantized Tensor.
    """
    if axis not in (0, -1):
        raise ValueError("axis parameter must be 0 (first axis) or -1 (last axis)")
    if qtype.bits == 8:
        if shift is not None:
            raise ValueError("shift cannot be specified for 8-bit qtypes")
        if group_size is not None:
            raise ValueError("group_size cannot be specified for 8-bit qtypes.")
        if axis is not None and t.shape[axis] == 1:
            # Quantizing along an axis of dimension 1 means quantizing per-tensor
            axis = None
        return WeightQBytesTensor.quantize(t, qtype, axis, scale, activation_qtype, optimized)
    if shift is None:
        raise ValueError("shift must be specified for qtypes lower than 8-bit")
    return WeightQBitsTensor.quantize(t, qtype, axis, group_size, scale, shift, optimized)
