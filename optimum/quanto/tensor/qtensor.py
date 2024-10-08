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
from torch.utils import _pytree as pytree


__all__ = ["QTensor", "qfallback"]


def qfallback(callable, *args, **kwargs):
    """Fallback method for QTensor inputs.

    When a torch function or an aten operation is not supported for the specified
    QTensor arguments, each QTensor arg or kwarg is dequantized to a torch.Tensor
    before calling the target function or op.
    """
    args, kwargs = pytree.tree_map_only(QTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


class QTensor(torch.Tensor):
    def __init__(self, qtype, axis):
        self._qtype = qtype
        self._axis = axis

    def dequantize(self):
        raise NotImplementedError

    def save_to_state_dict(self, destination, prefix, keep_vars):
        def serialize_tensor_subclass(t, destination, prefix, keep_vars):
            inner_tensors, meta = t.__tensor_flatten__()
            for name in inner_tensors:
                inner_tensor = getattr(t, name)
                if type(inner_tensor) is torch.Tensor:
                    # Leaf Tensor, we can serialize it
                    destination[prefix + name] = inner_tensor if keep_vars else inner_tensor.detach()
                else:
                    # Flatten also this inner Tensor
                    serialize_tensor_subclass(inner_tensor, destination, prefix + name + ".", keep_vars)

        # Recursively flatten QTensor into individual tensors
        serialize_tensor_subclass(self, destination, prefix, keep_vars)

    @property
    def axis(self):
        return self._axis

    @property
    def qtype(self):
        return self._qtype

    def numpy(self):
        return self.dequantize().cpu().numpy()

    def equal(self, other):
        if type(self) is not type(other):
            return False
        self_tensors, self_meta = self.__tensor_flatten__()
        _, other_meta = other.__tensor_flatten__()
        for name, value in self_meta.items():
            if other_meta[name] != value:
                return False
        for name in self_tensors:
            self_t = getattr(self, name)
            other_t = getattr(other, name)
            if self_t.device.type == "cpu" and self_t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # torch.equal is not implemented on CPU for float8 types
                if self_t.dtype != other_t.dtype:
                    return False
                if not torch.equal(self_t.to(torch.float32), other_t.to(torch.float32)):
                    return False
            elif not torch.equal(self_t, other_t):
                return False
        return True
