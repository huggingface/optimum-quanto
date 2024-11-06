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
from torch.overrides import TorchFunctionMode

from .optimizers import AbsmaxOptimizer, MaxOptimizer, SymmetricOptimizer
from .qtensor import QTensor
from .qtype import qint2, qint4, qtype
from .weights import quantize_weight


__all__ = ["QDynamicTensor"]


class QDynamicTensor(TorchFunctionMode):
    """A custom torch dispatch mode that uses dynamically quantized tensors.

    Args:
        tensor (`torch.Tensor`): the torch.Tensor that will be dynamically quantized
        qtype (`qtype`): the qtype to use to quantize the Tensor.
        axis (`int`): the quantization axis.
        optimizer (`Optimizer`): the optimizer to use to get the quantization parameters.
    """

    def __init__(self, tensor: torch.Tensor, qtype: qtype, axis: int, optimizer=None):
        super().__init__()
        assert not isinstance(tensor, QTensor)
        self.tensor = tensor
        self.qtype = qtype
        self.axis = axis
        self.group_size = None
        if qtype in (qint2, qint4):
            axis_dim = tensor.shape[axis]
            other_dim = tensor.numel() // axis_dim
            group_size = 128
            if other_dim > group_size:
                while other_dim % group_size != 0 and group_size > 32:
                    group_size -= 32
                if other_dim % group_size == 0:
                    self.group_size = group_size
        if optimizer is None:
            optimizer = AbsmaxOptimizer() if qtype.bits == 8 else MaxOptimizer()
        self.optimizer = optimizer

    def qtensor(self, other_qtype: qtype = None):
        """Return the dynamically quantized QTensor
        """
        # Quantize dynamically the tensor per-axis
        if isinstance(self.optimizer, SymmetricOptimizer):
            scale = self.optimizer(self.tensor, qtype=self.qtype, axis=self.axis)
            shift = None
        else:
            scale, shift = self.optimizer(
                self.tensor, qtype=self.qtype, axis=self.axis, group_size=self.group_size
            )
        return quantize_weight(
            self.tensor,
            qtype=self.qtype,
            axis=self.axis,
            scale=scale,
            shift=shift,
            group_size=self.group_size,
            activation_qtype=other_qtype,
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        other_qtype = None
        new_args = []
        for arg in args:
            new_arg = arg
            if isinstance(arg, QTensor):
                if other_qtype is None:
                    other_qtype = arg.qtype
                else:
                    assert arg.qtype == other_qtype
            else:
                qtag = getattr(arg, "qtag", None)
                if qtag == self.tensor.qtag:
                    # Replace the tensor by its dynamically quantized version
                    new_arg = self.qtensor(other_qtype)
            new_args.append(new_arg)
        return func(*new_args, **kwargs)

    def __enter__(self):
        super().__enter__()
        # Tag the target Tensor to identify it when dispatching
        self.tensor.qtag = id(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        # Untag the target Tensor
        delattr(self.tensor, "qtag")
