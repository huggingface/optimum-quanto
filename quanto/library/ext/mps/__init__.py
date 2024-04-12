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

import os

import torch
from torch.library import impl
from torch.utils.cpp_extension import load


__all__ = []


_ext = None


def ext():
    """Helper to load the MPS extension only when it is required"""
    global _ext
    if _ext is None:
        module_path = os.path.dirname(__file__)
        _ext = load(
            name="quanto_mps",
            sources=[f"{module_path}/unpack.mm", f"{module_path}/pybind_module.cpp"],
            extra_cflags=["-std=c++17"],
        )
    return _ext


@impl("quanto_ext::unpack", "MPS")
def unpack_mps(t: torch.Tensor, bits: int):
    return ext().unpack(t, bits)
