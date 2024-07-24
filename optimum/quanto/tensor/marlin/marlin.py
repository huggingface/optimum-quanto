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

from ..qbytes import QBytesTensor
from ..qtype import qtypes
from .packed import MarlinF8PackedTensor


class MarlinF8QBytesTensor(QBytesTensor):
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

        assert axis is None
        assert data.ndim == 2

        out_features = size[0]
        self._workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=data.device)

        if not isinstance(data, MarlinF8PackedTensor):
            scale = scale.repeat(1, out_features).to(data.device)
            data_packed = MarlinF8PackedTensor.pack(data)  # pack fp8 data to in32, and apply marlier re-ordering.
        else:
            # When freezing (`model.freeze()`), the data is already a MarlinF8PackedTensor.
            data_packed = data

        super().__init__(qtype, axis, size, stride, data_packed, scale)

    def dequantize(self):
        float8_data = self._data.unpack()
        return float8_data.to(self._scale.dtype) * self._scale.T

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_workspace"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 3
        assert len(meta) == 4
        data, scale = inner_tensors["_data"], inner_tensors["_scale"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return QBytesTensor(qtype, axis, size, stride, data, scale)
