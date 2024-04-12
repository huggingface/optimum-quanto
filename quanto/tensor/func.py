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

from functools import partial

import torch

from .qtensor import qfallback


__all__ = ["get_qtensor_func", "register_qtensor_func"]


_QTENSOR_FUNC_TABLE = {}


def register_qtensor_func(funcs):
    """
    Used for registering a new __torch_dispatch__ function to QTensor.

    The code to register a new function looks like:

    @register_qtensor_func(list_of_funcs)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(qfunc):
        for func in funcs:
            _QTENSOR_FUNC_TABLE[func] = partial(qfunc, func)

    return wrapper


def get_qtensor_func(func):
    return _QTENSOR_FUNC_TABLE.get(func, None)


@register_qtensor_func([torch._has_compatible_shallow_copy_type])
def has_compatible_shallow_copy_type(func, input: torch.Tensor, from_: torch.Tensor):
    # Prevent torch from trying to shallow copy one QTensor to another
    return False


@register_qtensor_func(
    [
        torch.nn.functional.cross_entropy,
        torch.nn.functional.cosine_similarity,
        torch.nn.functional.layer_norm,
        torch.nn.functional.log_softmax,
        torch.topk,
    ]
)
def unsupported_op(func, *args, **kwargs):
    return qfallback(func, *args, **kwargs)


@register_qtensor_func([torch.nn.functional.linear])
def linear(func, input, other, bias=None):
    output = torch.matmul(input, other.t())
    if bias is not None:
        output = output + bias
    return output
