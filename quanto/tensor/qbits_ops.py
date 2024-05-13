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

from copy import copy
from functools import partial
from typing import Callable, List

import torch

from .qbits import QBitsTensor


__all__ = ["get_qbitstensor_op_dispatch", "register_qbitstensor_op"]


_QBITSTENSOR_OP_TABLE = {}


def register_qbitstensor_op(aten_ops: List[Callable]):
    """
    Used for registering a new __torch_dispatch__ aten operation to QBitsTensor.

    The code to register a new operation looks like:

    @register_qbitstensor_op(list_of_ops)
    def foo(op, *args, **kwargs):
        <implementation>
    """

    def wrapper(op):
        for aten_op in aten_ops:
            _QBITSTENSOR_OP_TABLE[aten_op] = partial(op, aten_op)

    return wrapper


def get_qbitstensor_op_dispatch(aten_op):
    return _QBITSTENSOR_OP_TABLE.get(aten_op, None)


@register_qbitstensor_op([torch.ops.aten._to_copy])
def _to_copy(op, t, dtype=None, **kwargs):
    # Copy scale
    scale = op(t._scale, **kwargs)
    # Move data and zeropoint, ignoring dtype (it only applies to scale)
    data_kwargs = copy(kwargs)
    data_kwargs["dtype"] = torch.uint8
    data = op(t._data, **data_kwargs)
    zeropoint_kwargs = copy(kwargs)
    zeropoint_kwargs["dtype"] = torch.int8
    zeropoint = op(t._zeropoint, **data_kwargs)
    return QBitsTensor(t._qtype, t._axis, t._group_size, t.size(), t.stride(), data, scale, zeropoint)


@register_qbitstensor_op([torch.ops.aten.detach])
def detach(op, t):
    # Detach is required when copying and deserializing
    data = op(t._data)
    scale = op(t._scale)
    zeropoint = op(t._zeropoint)
    return QBitsTensor(t._qtype, t._axis, t._group_size, t.size(), t.stride(), data, scale, zeropoint)
