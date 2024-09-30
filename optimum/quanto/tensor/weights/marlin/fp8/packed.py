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


def pack_fp8_as_int32(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements).
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn

    if fp8_tensor.shape[0] % 4 != 0:
        raise ValueError(f"Leading tensor dimension is not divisable by 4: {fp8_tensor.shape[0]}")

    # Reshape to prepare for packing
    reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])

    # Convert fp8 to uint8 (byte) representation
    byte_tensor = reshaped.view(torch.uint8)

    # Pack 4 uint8 values into one int32
    packed = torch.zeros(
        fp8_tensor.shape[0] // 4,
        fp8_tensor.shape[1],
        dtype=torch.int32,
        device=fp8_tensor.device,
    )

    for i in range(4):
        packed.bitwise_or_(byte_tensor[:, i].to(torch.int32) << i * 8)

    return packed


def unpack_int32_to_fp8(int32_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reinterpret a tensor (a, b) of type int32 to a tensor (a * 4, b) of type float8_e4m3fn.
    """
    bits = 8

    unpacked = []
    # Unpack each set of values independently
    for i in range(4):
        mask = 2 ** (bits * (i + 1)) - 1
        tmp = (int32_tensor & mask) >> bits * i
        tmp = tmp.to(torch.uint8)
        unpacked.append(tmp)

    # Return the concatenated unpacked tensors
    unpacked = torch.cat(unpacked).view(torch.float8_e4m3fn)

    return unpacked


def get_scale_perms() -> torch.Tensor:
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return torch.tensor(scale_perm_single, dtype=torch.int64)


def get_row_permutation(n_rows: int) -> torch.Tensor:
    """
    Generates a tensor of shape (4 * n_rows,) giving the rows mapping to map from marlin-repacked weights to natural order.

    Example: if n_rows = 8, the row mapping from natural to marlin format is
    rows_idx = [0,  2,  4,  6,
                16, 18, 20, 22,
                8, 10, 12, 14,
                24, 26, 28, 30,
                1,  3,  5,  7,
                17, 19, 21, 23,
                9, 11, 13, 15,
                25, 27, 29, 31].
    """
    modulo = n_rows // 4 * 16 - 8
    b = n_rows // 2

    # Group by 16*k, then by 8 + 16*k
    rows_idx = [(i * 16) % modulo for i in range(b)]
    rows_idx[-1] = rows_idx[-2] + 16 if b > 2 else 8
    rows_idx = torch.tensor(rows_idx)

    # All even indexes, and then all odd indexes.
    rows_idx = torch.cat((rows_idx, rows_idx + 1))

    # Indexes are grouped by four, each spaced by 2.
    rows_idx = torch.tile(rows_idx[:, None], (1, 4))
    rows_idx = rows_idx + torch.tensor([[0, 2, 4, 6]])

    rows_idx = rows_idx.reshape(-1)

    # `rows_idx` holds the mapping of natural rows to marlin rows, so inverse it.
    rows_idx_rev = torch.empty_like(rows_idx)
    rows_idx_rev[rows_idx] = torch.arange(len(rows_idx))

    return rows_idx_rev


def get_column_permutation(n_col: int) -> torch.Tensor:
    """
    Gets the column mapping to map from marlin-repacked weights to natural order.

    The natural order to marlin is: `8 * rest + frac` to `rest + 32 * frac`, by blocks of 256 values.
    """
    tile_size = 256
    n_blocks = n_col // tile_size

    a = torch.arange(tile_size)
    rest = a % 8
    frac = a // 8

    original_index = 32 * rest + frac

    original_index = torch.arange(n_blocks)[:, None] * 256 + original_index
    original_index = original_index.reshape(-1)

    # The mapping per-column is:
    #
    #      64   64   64   64      64   64   64   64       64   64   64   64
    # ------------------------------------------------------------------------
    # |    0    1    2    3  |    0    1    2    3   |    0    1    2    3   |
    # ------------------------------------------------------------------------
    #
    # Hence to retrieve column 0, 1, 2, 3 in order, we need to
    # shuffle the blocks of 64 values.
    original_index = original_index.reshape(4 * n_blocks, 64)

    # Generate a shuffling as e.g. [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11] for the above.
    tmp1 = torch.arange(4)
    tmp1 = tmp1.repeat(n_blocks, 1).T.reshape(-1)  # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    tmp2 = torch.arange(n_blocks) * 4
    tmp2 = tmp2.repeat(4)  # e.g. [0, 4, 8, 0, 4, 8, 0, 4, 8, 0, 4, 8]

    remap_col_index = tmp1 + tmp2

    original_index = original_index[remap_col_index]
    original_index = original_index.reshape(-1)

    return original_index


class MarlinF8PackedTensor(torch.Tensor):
    def __new__(cls, data, size, stride, requires_grad=False):
        assert data.device.type == "cuda"
        assert data.dtype == torch.int32
        assert requires_grad is False
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=torch.int32, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, data, size, stride, requires_grad=False):
        self._data = data

    def __repr__(self):
        return f"MarlinF8PackedTensor({self._data})"

    @classmethod
    def pack(cls, tensor: torch.Tensor):
        out_features, in_features = tensor.shape

        data_int32 = pack_fp8_as_int32(tensor.T)  # pack fp8 data to in32.

        perm = torch.empty(0, dtype=torch.int, device=tensor.device)

        data_int32 = torch.ops.quanto.pack_fp8_marlin(
            b_q_weight=data_int32, perm=perm, size_k=in_features, size_n=out_features, num_bits=8
        )

        return cls(data_int32, size=tensor.size(), stride=tensor.stride())

    def unpack(self) -> torch.Tensor:
        """
        Reinterprets the packed tensor (a, b) of type int32 and in the marlin order, to a tensor (a * 4, b) of type float8_e4m3fn, in the natural order.
        """
        float8_data = unpack_int32_to_fp8(self._data)

        # complex indexing is not implemented for 'Float8_e4m3fn'
        uint8_data = float8_data.view(torch.uint8)

        n_rows, n_col = uint8_data.shape

        # swap columns
        column_map = get_column_permutation(n_col=n_col)

        uint8_data = uint8_data.T.contiguous()
        uint8_data = uint8_data[column_map]
        uint8_data = uint8_data.T.contiguous()

        uint8_data = uint8_data.reshape(uint8_data.shape[0] * 4, -1)

        # swap rows
        row_map = get_row_permutation(n_rows=n_rows)

        uint8_data = uint8_data[row_map]

        float8_data = uint8_data.view(torch.float8_e4m3fn)
        float8_data = float8_data.T  # As we originally transposed in `pack_fp8_as_int32`

        return float8_data

    @property
    def dtype(self):
        return torch.int32

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
        return MarlinF8PackedTensor(data, size, stride)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        # Convert back to tensor before calling any operation except detach and move
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            return cls(data, t.size(), t.stride())
        elif op.overloadpacket in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.get("dtype", torch.int32)
            if dtype != torch.int32:
                raise ValueError(f"MarlinF8PackedTensor are torch.int32 only and cannot be moved to {dtype}.")
            device = kwargs.get("device", t.device)
            if device.type == "cuda":
                data_kwargs = copy(kwargs)
                data_kwargs["dtype"] = t._data.dtype
                data = op(t._data, **data_kwargs)
                return cls(data, t.size(), t.stride())
            else:
                return t.unpack().to(device)
        else:
            args, kwargs = pytree.tree_map_only(cls, lambda x: x.unpack(), (args, kwargs or {}))
            return op(*args, **kwargs)
