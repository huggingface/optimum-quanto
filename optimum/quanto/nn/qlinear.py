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

from ..tensor import Optimizer, QBytesTensor, qtype
from ..tensor.qbits.awq.qbits import AWQBitsTensor
from ..tensor.qbits.tinygemm.qbits import TinyGemmQBitsTensor
from ..tensor.weights.marlin import MarlinF8QBytesTensor
from .qmodule import QModuleMixin, register_qmodule


__all__ = ["QLinear"]


def _forward_linear(input, other, bias):
    if isinstance(other, AWQBitsTensor):
        if type(input) is not torch.Tensor:
            input = input.dequantize()
        out_features, in_features = other.shape
        rows = input.numel() // in_features
        output = torch.ops.quanto.gemm(
            input,
            other._data._data,
            other._scale,
            other._shift,
            rows=rows,
            out_cols=out_features,
            in_cols=in_features,
            bits=4,
            group_size=other._group_size,
        )
    elif isinstance(other, TinyGemmQBitsTensor):
        if type(input) is not torch.Tensor:
            input = input.dequantize()
        in_features = input.shape[-1]
        out_features = other.shape[0]
        output_shape = input.shape[:-1] + (out_features,)
        output = torch._weight_int4pack_mm(
            input.view(-1, in_features), other._data._data, other._group_size, other._scale_shift
        )
        output = output.view(output_shape)
    elif isinstance(other, MarlinF8QBytesTensor):
        input_shape = input.shape

        if input.ndim > 2:
            input = input.view(-1, input_shape[-1])

        output = torch.ops.quanto_ext.fp8_marlin_gemm(
            input,
            b_q_weight=other._data._data,
            b_scales=other._scale,  # .to(input.dtype)
            workspace=other._workspace,
            num_bits=8,
            size_m=input.shape[0],
            size_n=other._scale.shape[1],
            size_k=input.shape[1],
        )

        if len(input_shape) > 2:
            output = output.reshape(input_shape[:-1] + (other._scale.shape[1],))
    elif isinstance(other, QBytesTensor):
        if input.__class__.__name__ == "QBytesTensor":
            output = torch.ops.quanto.qbytes_mm(input._data, other._data, input._scale * other._scale)
        else:
            output = torch.ops.quanto.qbytes_mm(input, other._data, other._scale)
    else:
        output = torch.matmul(input, other.t())
    if bias is not None:
        output = output + bias
    return output


class QLinearFunction(torch.autograd.Function):
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
    def forward(ctx, input, other, bias):
        ctx.save_for_backward(input, other)
        return _forward_linear(input, other, bias)

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


@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(
        cls, module, weights: qtype, activations: Optional[qtype] = None, optimizer: Optional[Optimizer] = None
    ):
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.requires_grad:
            return QLinearFunction.apply(input, self.qweight, self.bias)
        else:
            return _forward_linear(input, self.qweight, bias=self.bias)
