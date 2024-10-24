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
from enum import Enum

import numpy as np
import torch
from torch.utils import _pytree as pytree

from ..packing import unpack_int32_to_uint8


__all__ = ["AWQPackedTensor", "AWQPacking"]


AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def pack(unpacked: torch.Tensor, reorder=False):
    """
    Pack uint4 weights in an int32 tensor as expected by AWQ mixed mm kernel

    As compared to the standard packing, this adds an optional permutation of the columns
    for faster dequantization, as explained in "Who Says Elephants Can’t Run: Bringing Large
    Scale MoE Models into Cloud Scale Production", https://arxiv.org/pdf/2211.10017.

    Args:
        unpacked (`torch.Tensor`):
            The un-packed `torch.uint8` tensor
        reorder (`bool`):
            Whether columns should be reordered or not before packing.

    Returns:
        A int32 `torch.Tensor`.
    """
    bits = 4
    pack_num = 32 // bits
    packed = torch.zeros(unpacked.shape[0], unpacked.shape[1] // pack_num, dtype=torch.int32, device=unpacked.device)
    for col in range(unpacked.shape[1] // pack_num):
        if reorder:
            order_map = AWQ_ORDER
        else:
            order_map = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(pack_num):
            packed_col = unpacked[:, col * pack_num + order_map[i]].to(torch.int32)
            packed[:, col] |= packed_col << (i * bits)
    return packed


def reverse_awq_order(t: torch.Tensor):
    bits = 4
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.reshape(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.reshape(-1)

    t = t[:, reverse_order_tensor]

    return t


def unpack(packed: torch.Tensor, reorder=False):
    """Unpack a packed int32 tensor to a larger uint8 tensor

    Applies pack operations in reverse order (see pack method for details).

    Args:
        packed (`torch.Tensor`):
            The packed `torch.int32` tensor
        reorder (`bool`):
            Whether columns should be reordered or not.

    Returns:
        An unpacked uint8 `torch.Tensor` expanded along the second dimension.
    """
    unpacked = unpack_int32_to_uint8(packed, bits=4)
    if reorder:
        unpacked = reverse_awq_order(unpacked)
    return unpacked


def pack_v2(unpacked: torch.Tensor) -> torch.Tensor:
    """
    Pack uint4 weights in an int16 tensor as expected by AWQ second generation mixed mm kernel

    As compared to the standard packing, this adds three specific formatting:

    - permute rows to counter implicit permutation on Turing and Ampere architecture,
    - permute rows for faster dequantization,
    - interleave groups of 'interleave' rows for efficient parallel processing.

    Note that this formatting expects a group size of 128.

    Args:
        unpacked (`torch.Tensor`):
            The un-packed `torch.uint8` tensor

    Returns:
        A int16 `torch.Tensor`.
    """
    assert unpacked.device.type == "cuda"
    assert unpacked.ndim == 2
    N, K = unpacked.shape
    # These two values are hard-coded in the optimized kernels:
    # - I represents the 'interleave', i.e. the number of values packed at a single coordinate (16 bits / 4 bits),
    # - S represents the 'kernel stride', and is related to the group size (TBC).
    I = 4
    S = 64

    # 1. For faster dequantization, the tensor rows must be permuted as explained in:
    # https://github.com/NVIDIA/TensorRT-LLM/blob/035b99e0d09d4f2dfdb949810cf7245112aa4165/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp#L161
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...] => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    packed = unpacked.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 3, 2, 4)

    # Reorder each 8 weights for fast dequantization
    # From: "Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production"
    # https://arxiv.org/pdf/2211.10017
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    packed = packed.permute(0, 1, 2, 4, 3)
    packed = packed.reshape(N, K)

    # 2. For efficient parallelization, the rows are grouped and interleaved by blocks of kstride into a single row, as explained in:
    # https://github.com/NVIDIA/TensorRT-LLM/blob/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h#L69
    # interleaving (N, K) -> (N // I, I, K // S, S)
    packed = packed.reshape(N // I, I, K // S, S)
    # transpose (N // I, I, K // S, S) -> (N // I, K // S, I, S)
    packed = packed.permute(0, 2, 1, 3)
    # reshape (N // I, K // S, I, S) -> (N // I, K // S, S, I)
    packed = packed.reshape(N // I, K // S, S, I)
    # Packing (N // I, K // S, S, I) -> (N // I, K // S, S)
    packed = packed.to(torch.int32)
    packed = packed[..., 0] | (packed[..., 1] << 4) | (packed[..., 2] << 8) | (packed[..., 3] << 12)
    # Reshape to (N // I, K // S, S) -> (N // I, K)
    packed = packed.reshape(N // I, K)
    return packed.to(torch.int16).contiguous()


def unpack_v2(packed):
    """Unpack a packed int16 tensor to a larger uint8 tensor

    Applies pack operations in reverse order (see pack_v2 method for details).
    Warning: very slow, to be used for debug only.

    Args:
        packed (`torch.Tensor`):
            The packed `torch.int16` tensor

    Returns:
        An unpacked uint8 `torch.Tensor` expanded along the first dimension.
    """
    assert packed.device.type == "cuda"
    assert packed.ndim == 2
    I = 4
    S = 64
    N_div_I, K = packed.shape
    N = N_div_I * I
    # Reshape (N // I, K) -> (N // I, K // S, S, 1)
    unpacked = packed.reshape(N // I, K // S, S, 1)
    # Convert to uint16 (through numpy because not supported by pytorch)
    unpacked = unpacked.cpu().numpy().astype(np.uint16)
    # Unpack (N // I, K, S) -> (N // I, K // S, S, I)
    unpacked = torch.cat(
        [
            torch.tensor((unpacked & 0xF).astype(np.uint8)).to(packed.device),
            torch.tensor(((unpacked & 0xF0) >> 4).astype(np.uint8)).to(packed.device),
            torch.tensor(((unpacked & 0xF00) >> 8).astype(np.uint8)).to(packed.device),
            torch.tensor(((unpacked & 0xF000) >> 12).astype(np.uint8)).to(packed.device),
        ],
        axis=-1,
    )
    # reshape (N // I, K // S, S, I) -> (N // I, K // S, I, S)
    unpacked = unpacked.reshape(N // I, K // S, I, S)
    # transpose (N // I, K // S, I, S) -> (N // I, I, K // S, S)
    unpacked = unpacked.permute(0, 2, 1, 3)
    # deinterleaving (N // I, I, K // S, S) -> (N, K)
    unpacked = unpacked.reshape(N, K)

    # Final steps to reorder (see packing code for explaination)
    unpacked = unpacked.reshape(N, K // 32, 4, 2, 4).permute(0, 1, 2, 4, 3)
    unpacked = unpacked.permute(0, 1, 3, 2, 4)
    unpacked = unpacked.reshape(N, K)

    return unpacked


class AWQPacking(Enum):
    V1 = 1
    V2 = 2


class AWQPackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, packing, reorder, size, stride, requires_grad=False):
        # AWQPackedTensor represents uint8 data and can therefore NEVER require gradient
        assert data.device.type == "cuda"
        assert data.dtype == torch.int32 if packing == AWQPacking.V1 else torch.int16
        assert requires_grad is False
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=torch.uint8, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, data, packing, reorder, size, stride, requires_grad=False):
        self._data = data
        self._packing = packing
        self._reorder = reorder

    def __repr__(self):
        return f"AWQPackedTensor({self._data}, packing={self._packing}, reorder={self._reorder})"

    @classmethod
    def pack(cls, t, packing=AWQPacking.V1, reorder=False):
        if packing == AWQPacking.V1:
            data = pack(t, reorder=reorder)
        else:
            data = pack_v2(t)
        # We need to store size and stride to make sure the unpacked data has the correct shape
        return AWQPackedTensor(data, packing, reorder, t.size(), t.stride())

    def unpack(self):
        if self._packing == AWQPacking.V1:
            return unpack(self._data, self._reorder)
        return unpack_v2(self._data)

    @property
    def dtype(self):
        return torch.uint8

    def __tensor_flatten__(self):
        inner_tensors = ["_data"]
        # Since meta can be used for serialization, use only AST compatible strings
        meta = {
            "packing": str(self._packing),
            "reorder": str(self._reorder),
            "size": str(list(self.size())),
            "stride": str(self.stride()),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 1
        assert len(meta) == 4
        data = inner_tensors["_data"]
        # Meta should contain only AST compatible strings
        packing = ast.literal_eval(meta["packing"])
        reorder = ast.literal_eval(meta["reorder"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return AWQPackedTensor(data, packing, reorder, size, stride)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Convert back to tensor before calling any operation except detach and move
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            return AWQPackedTensor(data, t._packing, t._reorder, t.size(), t.stride())
        elif op.overloadpacket in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.get("dtype", torch.uint8)
            if dtype != torch.uint8:
                raise ValueError(f"AWQPackedTensor are torch.uint8 only and cannot be moved to {dtype}.")
            device = kwargs.get("device", t.device)
            # AWQPackedTensor can only be moved to CUDA devices
            if device.type == "cuda":
                data_kwargs = copy(kwargs)
                data_kwargs["dtype"] = t._data.dtype
                data = op(t._data, **data_kwargs)
                return AWQPackedTensor(data, t._packing, t._reorder, t.size(), t.stride())
        args, kwargs = pytree.tree_map_only(AWQPackedTensor, lambda x: x.unpack(), (args, kwargs or {}))
        return op(*args, **kwargs)

    def numpy(self):
        return self.unpack().cpu().numpy()
