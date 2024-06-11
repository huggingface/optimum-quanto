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

from .optimizers import AbsmaxOptimizer, AffineOptimizer, MaxOptimizer, Optimizer, SymmetricOptimizer
from .qtype import qtype
from .quantizers import AffineQuantizer, SymmetricQuantizer


__all__ = ["quantize_weight"]


default_affine_optimizer = MaxOptimizer()
default_symmetric_optimizer = AbsmaxOptimizer()


def quantize_weight(
    t: torch.Tensor, qtype: qtype, axis: int, group_size: Optional[int] = None, optimizer: Optional[Optimizer] = None
):
    """Quantize a weight Tensor.

    Weights are always quantized per-axis.

    Args:
        t (`torch.Tensor`): the weight Tensor to quantize
        qtype (`quanto.qtype`): The target quantization type
        axis ('int`): The quantization axis (0 or -1)
        group_size (`Optional[int]`): The quantization group size
        optimizer (`Optional[quanto.Optimizer]`): An optimizer to evaluate the scale if not provided.
            Defaults to a max Optimizer.

    Returns:
        A quantized Tensor.
    """
    if axis not in (0, -1):
        raise ValueError("axis parameter must be 0 (first axis) or -1 (last axis)")
    if qtype.bits == 8:
        if optimizer is None:
            optimizer = default_symmetric_optimizer
        else:
            if not isinstance(optimizer, SymmetricOptimizer):
                raise ValueError("A SymmetricOptimizer is expected")
        if group_size is not None:
            raise ValueError("group_size cannot be specified for 8-bit qtypes.")
        if axis is not None and t.shape[axis] == 1:
            # Quantizing along an axis of dimension 1 means quantizing per-tensor
            axis = None
        scale = optimizer(t, qtype.bits, axis)
        return SymmetricQuantizer.apply(t, qtype, axis, scale)
    if optimizer is None:
        optimizer = default_affine_optimizer
    else:
        if not isinstance(optimizer, AffineOptimizer):
            raise ValueError("An AffineOptimizer is expected")
    scale, zeropoint = optimizer(t, qtype.bits, axis, group_size)
    return AffineQuantizer.apply(t, qtype, axis, group_size, scale, zeropoint)
