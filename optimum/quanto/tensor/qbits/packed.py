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

import torch
from torch.utils import _pytree as pytree


__all__ = ["PackedTensor"]


def pack_weights(intweights: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack int4 / int2 weights in a uint8 tensor

    What packing means? Assume we have 4 values that are in 2bit but encoded in 8bit
    (because torch does not have native support for 2-bit datatypes)

    > 0000 0011 | 0000 0010 | 0000 0001 | 0000 0000

    We can pack them in a single 8-bit uint value

    > 1110 0100

    Therefore instead of saving 4 values in 8-bit precision we save a single value of 8-bit precision saving 24 bits in total.

    Args:
        intweights (`torch.Tensor`):
            The un-packed `torch.uint8` tensor
        bits (`int`):
            The actual `bits` - can be 2, 4
    """
    original_shape = intweights.shape
    values_per_item = 8 // bits
    row_dim = (original_shape[0] + values_per_item - 1) // values_per_item

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    packed = torch.zeros(packed_tensor_shape, device=intweights.device, dtype=torch.uint8)
    unpacked = intweights.to(torch.uint8)

    def lshift(t: torch.Tensor, bits: int):
        if t.device.type == "mps":
            # lshift is not supported on MPS device
            return t * (2**bits)
        return t << bits

    it = min(values_per_item, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        packed[: (end - start)] |= lshift(unpacked[start:end], bits * i)

    return packed


class PackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, bits, size, stride, requires_grad=False):
        # PackedTensor represents uint8 data and can therefore NEVER require gradient
        assert data.dtype == torch.uint8
        assert requires_grad is False
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=torch.uint8, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, data, bits, size, stride, requires_grad=False):
        self._bits = bits
        self._data = data

    def __repr__(self):
        autograd_info = (
            f", grad_fn={self.grad_fn}" if self.grad_fn else ", requires_grad=True" if self.requires_grad else ""
        )
        return f"PackedTensor({self._data}, bits={self._bits}, public_dtype={self.dtype}{autograd_info})"

    @classmethod
    def pack(cls, t, bits=4):
        assert bits in (2, 4)
        assert t.dtype == torch.uint8
        data = pack_weights(t, bits)
        # We need to store size and stride to make sure the unpacked data has the correct shape
        return PackedTensor(data, bits, t.size(), t.stride())

    def unpack(self):
        unpacked_data = torch.ops.quanto.unpack(self._data, self._bits)
        # Adjust the first dimension, as unpacked data may have extra rows if the original shape is not a multiple of 8 // bits
        return unpacked_data[: self.shape[0]]

    @property
    def bits(self):
        return self._bits

    @property
    def dtype(self):
        return torch.uint8

    @staticmethod
    def load_from_state_dict(state_dict, prefix, bits, size, stride, missing_keys):
        if prefix + "_data" not in state_dict:
            missing_keys.append(prefix + "_data")
            return

        inner_tensors_dict = {"_data": state_dict.pop(prefix + "_data")}
        meta = [name.replace(prefix, "") for name in state_dict.keys() if name.startswith(prefix)]
        meta = {"bits": str(bits), "size": str(list(size)), "stride": str(stride)}
        return PackedTensor.__tensor_unflatten__(inner_tensors_dict, meta, None, None)

    def __tensor_flatten__(self):
        inner_tensors = ["_data"]
        # Since meta can be used for serialization, use only AST compatible strings
        meta = {"bits": str(self._bits), "size": str(list(self.size())), "stride": str(self.stride())}
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 1
        assert len(meta) == 3
        data = inner_tensors["_data"]
        # Meta should contain only AST compatible strings
        bits = ast.literal_eval(meta["bits"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return PackedTensor(data, bits, size, stride)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Convert back to tensor before calling any operation except detach
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            return PackedTensor(data, t._bits, t.size(), t.stride())
        elif op.overloadpacket in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.get("dtype", torch.uint8)
            if dtype != torch.uint8:
                raise ValueError(f"PackedTensor are torch.uint8 only and cannot be moved to {dtype}.")
            # Move data
            data = op(t._data, **kwargs)
            return PackedTensor(data, t._bits, t.size(), t.stride())
        args, kwargs = pytree.tree_map_only(PackedTensor, lambda x: x.unpack(), (args, kwargs or {}))
        return op(*args, **kwargs)

    def numpy(self):
        return self.unpack().cpu().numpy()
