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


__all__ = ["QuantizedLinearFunction"]


class QuantizedLinearFunction(torch.autograd.Function):
    """Quantized linear function.

    This is a quantized implementation of torch.nn.functional.linear.

    It defines explicitly the backward pass instead of letting pytorch
    build it by combining the gradients of the underlying quantized operations.

    This has two main benefits:

    - this saves computations,
    - this allows to use operations that do not have a registered backward method,
    such as quanto custom operations.

    The drawback is that the extra tensors involved in the quantization graph, such as
    the scales and shift, cannot be trained.
    This is however consistent with the quanto quantizers backward pass, that returns
    a zero gradient for these tensors.
    """

    @staticmethod
    def forward(ctx, input, other, bias=None):
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
