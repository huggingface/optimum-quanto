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
                if type(inner_tensor) == torch.Tensor:
                    # Leaf Tensor, we can serialize it
                    destination[prefix + name] = inner_tensor if keep_vars else inner_tensor.detach()
                else:
                    # Flatten also this inner Tensor
                    serialize_tensor_subclass(inner_tensor, destination, prefix + name + ".", keep_vars)
            for name, value in meta.items():
                destination[prefix + name] = value

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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Dispatch torch functions applied on QTensor inputs

        This method is called whenever a torch function (such as `torch.nn.functional.linear`)
        is called with at least one QTensor parameter. It is shared by all QTensor subclasses.

        It looks for registered dispatched functions (see qtensor_func.py):
        - if a quantized implementation exists for the selected function, it is called,
        - otherwise, the original implementation is called, deactivating further functional dispatch.

        During the execution of the standard torch function, a second-level of dispatch will
        happen, but this time directly on individual torch Tensor operations (mainly ATEN).

        This second dispatch phase is specific to each QTensor subclass.
        """
        from .qtensor_func import get_qtensor_func

        kwargs = kwargs or {}

        # Look for a func accepting QTensor inputs
        qfunc = get_qtensor_func(func)
        if qfunc is not None:
            return qfunc(*args, **kwargs)
        # Defer to dispatcher to look instead for QTensor subclasses operations
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
