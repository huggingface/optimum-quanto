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

import timeit

import torch


def mm(a, b):
    return torch._int_mm(a, b)


A = torch.randint(1, 10, [2400, 2400]).type(torch.int8).cuda()
B = torch.randint(1, 10, [2400, 2400]).type(torch.int8).cuda()
it = 100

# Warmup (slow)
mm(A, B)
# Get a reference
print(timeit.Timer(lambda: mm(A, B)).timeit(it) / it)

cmm = torch.compile(mm, backend="inductor")
# First invocation will trigger the actual compilation
cmm(A, B)
# Now compare execution time
print(timeit.Timer(lambda: cmm(A, B)).timeit(it) / it)
