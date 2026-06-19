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

from . import qbytes_mm as _qbytes_mm  # noqa: F401
from . import quantize as _quantize  # noqa: F401


try:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
    from torch.distributed.tensor._ops.utils import register_prop_rule
except ImportError:
    DTensorSpec = None
    TensorMeta = None
    OpSchema = None
    OutputSharding = None
    register_prop_rule = None


def _get_quantize_symmetric_dtype(op_schema: OpSchema) -> torch.dtype:
    if "dtype" in op_schema.kwargs_schema:
        dtype = op_schema.kwargs_schema["dtype"]
    else:
        dtype = op_schema.args_schema[1]
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(dtype)}")
    return dtype


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride = [1]
    for i in range(1, len(shape)):
        stride.insert(0, stride[0] * shape[-i])
    return tuple(stride)


if register_prop_rule is not None:

    @register_prop_rule(torch.ops.quanto.quantize_symmetric.default)
    def quantize_symmetric_rule(op_schema: OpSchema) -> OutputSharding:
        base_spec = op_schema.args_schema[0]
        if not isinstance(base_spec, DTensorSpec):
            raise AssertionError
        if base_spec.tensor_meta is None:
            raise AssertionError

        tensor_meta = TensorMeta(
            base_spec.tensor_meta.shape,
            base_spec.tensor_meta.stride,
            _get_quantize_symmetric_dtype(op_schema),
        )
        return OutputSharding(
            DTensorSpec.from_dim_map(
                base_spec.mesh,
                base_spec.dim_map,
                base_spec.sums,
                tensor_meta=tensor_meta,
            )
        )

    @register_prop_rule(torch.ops.quanto.qbytes_mm.default)
    def qbytes_mm_rule(op_schema: OpSchema) -> OutputSharding:
        activation_spec, weight_spec, scale_spec = op_schema.args_schema
        if not isinstance(activation_spec, DTensorSpec):
            raise AssertionError
        if not isinstance(weight_spec, DTensorSpec):
            raise AssertionError
        if not isinstance(scale_spec, DTensorSpec):
            raise AssertionError
        if activation_spec.tensor_meta is None:
            raise AssertionError
        if weight_spec.tensor_meta is None:
            raise AssertionError
        if scale_spec.tensor_meta is None:
            raise AssertionError

        output_shape = torch.Size((*activation_spec.tensor_meta.shape[:-1], weight_spec.tensor_meta.shape[0]))
        output_dim_map = [*activation_spec.dim_map[:-1], weight_spec.dim_map[0]]
        tensor_meta = TensorMeta(
            output_shape,
            _contiguous_stride(output_shape),
            scale_spec.tensor_meta.dtype,
        )
        return OutputSharding(
            DTensorSpec.from_dim_map(
                activation_spec.mesh,
                output_dim_map,
                [*activation_spec.sums, *weight_spec.sums],
                tensor_meta=tensor_meta,
            )
        )
