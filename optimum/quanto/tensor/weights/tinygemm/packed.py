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

import ast
from copy import copy

import torch
from torch.utils import _pytree as pytree


__all__ = ["TinyGemmPackedTensor"]


class TinyGemmPackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, size, stride, requires_grad=False):
        # TinyGemmPackedTensor represents uint8 data and can therefore NEVER require gradient
        assert requires_grad is False
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=torch.uint8, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, data, size, stride, requires_grad=False):
        self._data = data

    def __repr__(self):
        return f"TinyGemmPackedTensor({self._data})"

    @classmethod
    def pack(cls, t):
        """Pack a torch.Tensor for tinygemm kernel

        This packs uint4 weights in an int32 tensor as expected by the torch tinygemm mixed mm kernel

        Args:
            t (`torch.Tensor`):
                The un-packed `torch.uint8` tensor

        Returns:
            A `TinyGemmPackedTensor`.
        """
        inner_ktiles = 2
        t = t.to(torch.int32).contiguous()
        if t.device.type == "cpu":
            data = torch._convert_weight_to_int4pack_for_cpu(t, innerKTiles=inner_ktiles)
        else:
            t_uint8 = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
            data = torch._convert_weight_to_int4pack(t_uint8, innerKTiles=inner_ktiles)
        # We need to store size and stride to make sure the unpacked data has the correct shape
        return TinyGemmPackedTensor(data, t.size(), t.stride())

    def unpack(self):
        """Unpack the packed tensor to a torch.Tensor

        Packing is device specific and implemented in undocumented dedicated kernels
        that are synchronized with the corresponding matrix multiplication operation.

        Instead of implementing a dedicated unpacking code, we pass an identity matrix
        to the mm operation with identity scale and shifts to produce the unpacked uint8 weights.

        Returns:
            An unpacked uint8 `torch.Tensor` expanded along the second dimension.
        """
        out_features, in_features = self.size()
        # We need to pass a group_size to the mm and format the scale and shift accordingly,
        # although it does not modify the calculation since we use identity scales and shifts.
        # We arbitrarily choose the smallest group_size to be sure it divides in_features
        group_size = 32
        scale_and_shift_shape = (in_features // group_size, out_features, 2)
        # Initialize identity scale
        id_scale_and_shift = torch.ones(scale_and_shift_shape, dtype=torch.bfloat16, device=self.device)
        # Set shift to mid-point, i.e. 2 **(bits - 1)
        id_scale_and_shift[:, :, 1] = 8

        identity = torch.eye(in_features, dtype=torch.bfloat16, device=self.device)
        if self._data.device.type == "cpu":
            unpacked_data = torch._weight_int4pack_mm_for_cpu(identity, self._data, group_size, id_scale_and_shift)
        else:
            unpacked_data = torch._weight_int4pack_mm(identity, self._data, group_size, id_scale_and_shift)

        return unpacked_data.t().to(torch.uint8)

    @property
    def dtype(self):
        return torch.uint8

    def __tensor_flatten__(self):
        inner_tensors = ["_data"]
        # Since meta can be used for serialization, use only AST compatible strings
        meta = {
            "size": str(list(self.size())),
            "stride": str(self.stride()),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 1
        assert len(meta) == 2
        data = inner_tensors["_data"]
        # Meta should contain only AST compatible strings
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return TinyGemmPackedTensor(data, size, stride)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Convert back to tensor before calling any operation except detach and move
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            return TinyGemmPackedTensor(data, t.size(), t.stride())
        elif op.overloadpacket in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.get("dtype", torch.uint8)
            if dtype != torch.uint8:
                raise ValueError(f"TinyGemmPackedTensor are torch.uint8 only and cannot be moved to {dtype}.")
            data_kwargs = copy(kwargs)
            data_kwargs["dtype"] = t._data.dtype
            if kwargs.get("device", t.device).type != t.device.type:
                # Packing is device specific, so we need to unpack before moving
                unpacked = t.unpack()
                unpacked = op(unpacked, **data_kwargs)
                return TinyGemmPackedTensor.pack(unpacked)
            # If we stay on the same device type, just copy/move packed data
            data = op(t._data, **data_kwargs)
            return TinyGemmPackedTensor(data, t.size(), t.stride())
        args, kwargs = pytree.tree_map_only(TinyGemmPackedTensor, lambda x: x.unpack(), (args, kwargs or {}))
        return op(*args, **kwargs)

    def numpy(self):
        return self.unpack().cpu().numpy()
