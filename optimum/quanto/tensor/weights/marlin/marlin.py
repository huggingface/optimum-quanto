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

from ...qtype import qtypes
from ..qbytes import WeightQBytesTensor
from .packed import MarlinF8PackedTensor, get_scale_perms


class MarlinF8QBytesTensor(WeightQBytesTensor):
    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, requires_grad=False):
        assert data.device.type == "cuda"
        assert data.device == scale.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, size, stride, data, scale, requires_grad=False):
        if requires_grad:
            raise NotImplementedError("Backward with Marlin FP8 is not implemented.")

        assert axis == 0
        assert data.ndim == 2

        out_features = size[0]
        self._workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=data.device)

        # TODO: Here we should use `not isinstance(data, MarlinF8PackedTensor)`, but `torch.compile` is bugged when using that.
        # Somewhere in the internals of torch.compile, `data` gets converted to a `torch._subclasses.fake_tensor.FakeTensor` not inheriting from `MarlinF8PackedTensor` and torch then goes into the wrong controlflow.
        # Reference: https://pytorch.slack.com/archives/C033H6DJSJU/p1721837684035049
        if data.dtype != torch.int32:
            assert scale.shape == (out_features, 1)
            scale_perm_single = get_scale_perms()
            scale = scale.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
            scale = scale.reshape(-1, out_features).contiguous()

            data_packed = MarlinF8PackedTensor.pack(data)  # pack fp8 data to in32, and apply marlier re-ordering.
        else:
            # When freezing (`model.freeze()`), the data is already a MarlinF8PackedTensor and scale is already repacked.
            data_packed = data

        super().__init__(qtype, axis, size, stride, data_packed, scale)

    def dequantize(self):
        float8_data = self._data.unpack()

        scale_perm_single = get_scale_perms()

        # `scale_perm_single` holds the mapping of natural to marlin, so inverse it here.
        scale_perm_single_rev = torch.empty_like(scale_perm_single)
        scale_perm_single_rev[scale_perm_single] = torch.arange(len(scale_perm_single))

        scale_reordered = self._scale.reshape((-1, len(scale_perm_single_rev)))[:, scale_perm_single_rev]
        scale_reordered = scale_reordered.reshape(-1, self._scale.shape[1]).contiguous()

        return float8_data.to(scale_reordered.dtype) * scale_reordered.T

    def __repr__(self):
        return f"MarlinF8QBytesTensor({self._data}, scale={self._scale}, dtype={self.dtype})"

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        assert len(meta) == 4
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return MarlinF8QBytesTensor(qtype, axis, size, stride, data, scale)
