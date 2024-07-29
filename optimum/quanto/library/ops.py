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

import warnings
from contextlib import contextmanager

import torch


if torch.cuda.is_available():
    from .extensions.cuda import ext

    # This is required to be able to access `torch.ops.quanto_ext.*` members defined in C++ through `TORCH_LIBRARY`.
    _ = ext.lib

# This file contains the definitions of all operations under torch.ops.quanto


_ext_enabled = True


@contextmanager
def disable_extensions():
    """Disable quanto extensions (debug)"""
    try:
        global _ext_enabled
        _ext_enabled = False
        yield
    finally:
        _ext_enabled = True


def define(name, schema):
    """Define a new quanto operation.

    The operation will actually be defined in three libraries:
    - the top-level quanto library as quanto::<op>,
    - the quanto python library as quanto_py::<op>,
    - the quanto extension library as quanto_ext::<op>.

    Only the implementations for the python and extension library need
    to be provided: the top-level implementation for the operation is
    provided when calling this method and simply routes the calls towards
    either the python or extension implementations based on the selected
    mode.
    """
    for libname in ["quanto", "quanto_py", "quanto_ext"]:
        torch.library.define(f"{libname}::{name}", schema)

    # Provide the inplementation for all dispatch keys in the main library
    @torch.library.impl(f"quanto::{name}", "default")
    def impl(*args, **kwargs):
        if _ext_enabled:
            try:
                return getattr(torch.ops.quanto_ext, name)(*args, **kwargs)
            except Exception as e:
                if isinstance(e, NotImplementedError):
                    message = f"No optimized kernel found for quanto::{name}."
                else:
                    message = f"An exception was raised while calling the optimized kernel for quanto::{name}: {e}"
                warnings.warn(message + " Falling back to default implementation.")
        return getattr(torch.ops.quanto_py, name)(*args, **kwargs)


define("unpack", "(Tensor self, int bits) -> Tensor")


@torch.library.impl_abstract("quanto_ext::fp8_marlin_gemm")
def fp8_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
):
    assert b_scales.dtype == torch.float16 or b_scales.dtype == torch.bfloat16
    assert b_q_weight.dim() == 2
    assert b_q_weight.dtype == torch.int32

    return torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)


@torch.library.impl_abstract("quanto_ext::gptq_marlin_repack")
def gptq_marlin_repack(
    b_q_weight: torch.Tensor, perm: torch.Tensor, size_k: torch.Tensor, size_n: torch.Tensor, num_bits: torch.Tensor
):
    assert b_q_weight.dim() == 2
    assert b_q_weight.dtype == torch.int32

    return torch.empty(
        (b_q_weight.shape[0] // 4, b_q_weight.shape[1] * 4), dtype=b_q_weight.dtype, device=b_q_weight.device
    )
