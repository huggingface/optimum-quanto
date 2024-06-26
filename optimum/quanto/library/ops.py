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
define(
    "gemm",
    "(Tensor input,"
    " Tensor other,"
    " Tensor other_scale,"
    " Tensor other_zeropoint,"
    " int rows,"
    " int out_cols,"
    " int in_cols,"
    " int bits,"
    " int group_size)"
    " -> Tensor",
)
