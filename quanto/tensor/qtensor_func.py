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


# Below is a list of functions that we always want to operate on dequantized inputs
# We therefore provide a dispatched method that does it explicitly.
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


class QTensorLinear(torch.autograd.Function):
    """Quantized linear function.

    This is a quantized implementation of torch.nn.functional.linear.

    It defines explicitly the backward pass instead of letting pytorch
    build it by combining the gradients of the underlying quantized operations.

    This has two main benefits:

    - this saves computations,
    - this allows to use operations that do not have a registered backward method,
    such as quanto custom operations.

    The drawback is that the extra tensors involved in the quantization graph, such as
    the scales and zeropoint, cannot be trained.
    This is however consistent with the quanto quantizers backward pass, that returns
    a zero gradient for these tensors.
    """

    @staticmethod
    def forward(ctx, input, other, bias):
        ctx.save_for_backward(input, other)
        output = torch.matmul(input, other.t())
        if bias is not None:
            output = output + bias
        return output

    def backward(ctx, gO):
        input_gO = other_gO = bias_gO = None
        input, other = ctx.saved_tensors
        out_features, in_features = other.shape
        if ctx.needs_input_grad[0]:
            # grad(A@(B.t()) = gO => grad(A) = gO@(B.t().t()) = gO@B
            input_gO = torch.matmul(gO, other)
        if ctx.needs_input_grad[1]:
            # grad(B@A.t()) = gO.t() => grad(B) = gO.t()@(A.t().t()) = gO.t()@A
            other_gO = torch.matmul(gO.view(-1, out_features).t(), input.view(-1, in_features))
        if ctx.needs_input_grad[2]:
            # Bias gradient is the sum on all dimensions but the last one
            dim = tuple(range(gO.ndim - 1))
            bias_gO = gO.sum(dim)
        return input_gO, other_gO, bias_gO


@register_qtensor_func([torch.nn.functional.linear])
def linear(func, input, other, bias=None):
    return QTensorLinear.apply(input, other, bias)
