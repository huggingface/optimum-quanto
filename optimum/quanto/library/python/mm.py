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


@torch.library.impl("quanto_py::dqmm", "default")
def dqmm(input: torch.Tensor, other: torch.Tensor, other_scale: torch.Tensor):
    if other.dtype.is_floating_point:
        # An explicit type promotion avoids errors with float8 tensors (but is slower with integer)
        pother = other.to(other_scale.dtype)
        return torch.ops.aten.mm(input, pother * other_scale)
    return torch.ops.aten.mm(input, other * other_scale)
