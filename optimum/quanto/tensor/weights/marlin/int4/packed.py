import ast
from copy import copy

import numpy as np
import torch
from torch.utils import _pytree as pytree

from ...packing import unpack_int32_to_uint8
from ...reordering import reorder, reverse


__all__ = ["MarlinInt4PackedTensor"]


# From: https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py#L40
# this func does 2 things
# 1. 1 thread can load 32 4bit == 128bit weights used for mulitple mma instructions at once
# 2. faster dequant via parallel half2 mul
def _get_perm():
    perm = []
    # 32 == # of threads in 1 warp
    for i in range(32):
        perm1 = []
        # column id in 16x8 weight block
        # check https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
        col = i // 4
        # 1 32bit (int32) == 8 4bit, 1 thread has 4 weights per 16x8 & 4bit weights are packed in int32, so needs 2 16x8 == 1 16x16 blocks
        for block in [0, 1]:
            # row id in 16x8 weight block
            # check https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                # 8 weights used for 1 thread (16x16 block) are contiguous in memory via interleaving
                # e.g. T0 uses (0, 16, 128, 144, 8, 24, 136, 152)
                perm1.append(16 * row + col + 8 * block)
        # 1 128bit (int4) == 4 32bit, 1 thread loads 128bit at once, so needs 4 16x16 == 1 16x64 blocks
        for j in range(4):
            # 32 weights loaded by 1 thread (16x64 block) are contiguous in memory via interleaving
            # e.g. T0 uses ((0 ~ 152) + 0 * 256, (0 ~ 152) + 1 * 256, ..., (0 ~ 152) + 3 * 256)
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    # for faster dequant
    # check https://arxiv.org/pdf/2211.10017
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


_perm = _get_perm()
_rev_perm = reverse(_perm)


# From: https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py#L102
def pack(unpacked: torch.Tensor):
    w = unpacked
    N, K = w.shape
    w = unpacked.t()
    # 16 == tile size, marlin uses 16x16 tile, so 16x16 grouping via interleaving
    w = w.reshape((K // 16, 16, N // 16, 16))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((K // 16, N * 16))
    res = w
    # _perm.numel() == 1024 == 4 16x16, permute weights with 4 16x16 unit for efficient mma + dequant
    res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
    p = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        p |= res[:, i::8] << 4 * i
    p = torch.from_numpy(p.astype(np.int32)).to(w.device)
    return p


def unpack(packed, orig_shape):
    N, K = orig_shape
    # Unpack to recover individual values
    unpacked = unpack_int32_to_uint8(packed, bits=4).to(torch.uint8)
    # Recover the original ordering
    unpacked = reorder(unpacked, _rev_perm)
    # Apply block permutations in the reverse order
    unpacked = unpacked.reshape(K // 16, N // 16, 16, 16)
    unpacked = unpacked.permute((0, 2, 1, 3))
    unpacked = unpacked.reshape(K, N)
    return unpacked.t()


class MarlinInt4PackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, size, stride, requires_grad=False):
        assert data.device.type == "cuda"
        assert data.dtype == torch.int32
        assert requires_grad is False
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=torch.uint8, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, data, size, stride, requires_grad=False):
        self._data = data

    def __repr__(self):
        return f"MarlinInt4PackedTensor({self._data})"

    @classmethod
    def pack(cls, t):
        data = pack(t)
        return MarlinInt4PackedTensor(data, t.size(), t.stride())

    def unpack(self):
        return unpack(self._data, self.size())

    @property
    def dtype(self):
        return torch.uint8

    def __tensor_flatten__(self):
        inner_tensors = ["_data"]
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
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return MarlinInt4PackedTensor(data, size, stride)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        if op.overloadpacket is torch.ops.aten.detach:
            t = args[0]
            data = op(t._data)
            return MarlinInt4PackedTensor(data, t.size(), t.stride())
        elif op.overloadpacket in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.get("dtype", torch.uint8)
            if dtype != torch.uint8:
                raise ValueError(f"MarlinInt4PackedTensor are torch.uint8 only and cannot be moved to {dtype}.")
            device = kwargs.get("device", t.device)
            if device.type == "cuda":
                data_kwargs = copy(kwargs)
                data_kwargs["dtype"] = t._data.dtype
                data = op(t._data, **data_kwargs)
                return MarlinInt4PackedTensor(data, t.size(), t.stride())
            return t.unpack()
        args, kwargs = pytree.tree_map_only(MarlinInt4PackedTensor, lambda x: x.unpack(), (args, kwargs or {}))
        return op(*args, **kwargs)

    def numpy(self):
        return self.unpack().cpu().numpy()
