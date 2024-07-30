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

import functools
import gc
import os
import unittest

import pytest
import torch
from packaging import version

from optimum.quanto import QBitsTensor, absmax_scale, qint8, quantize_activation, quantize_weight


# Used to test the hub
USER = "__DUMMY_TRANSFORMERS_USER__"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"

# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"


def torch_min_version(v):
    def torch_min_version_decorator(test):
        @functools.wraps(test)
        def test_wrapper(*args, **kwargs):
            if version.parse(torch.__version__) < version.parse(v):
                pytest.skip(f"Requires pytorch >= {v}")
            test(*args, **kwargs)

        return test_wrapper

    return torch_min_version_decorator


def device_eq(a, b):
    if a.type != b.type:
        return False
    a_index = a.index if a.index is not None else 0
    b_index = b.index if b.index is not None else 0
    return a_index == b_index


def random_tensor(shape, dtype=torch.float32, device="cpu"):
    if dtype.is_floating_point:
        rand_dtype = dtype if dtype.itemsize > 1 else torch.float16
        # Generate a random tensor between -1. and 1.
        t = torch.rand(shape, dtype=rand_dtype, device=device) * 2 - 1
        return t.to(dtype)
    else:
        assert dtype == torch.int8
        return torch.randint(-127, 127, shape, dtype=torch.int8, device=device)


def random_qactivation(shape, qtype=qint8, dtype=torch.float32, device="cpu"):
    t = random_tensor(shape, dtype, device=device)
    scale = absmax_scale(t, qtype=qtype)
    return quantize_activation(t, qtype=qtype, scale=scale)


def random_qweight(shape, qtype, dtype=torch.float32, axis=0, group_size=None, device="cpu"):
    t = random_tensor(shape, dtype, device=device)
    return quantize_weight(t, qtype=qtype, axis=axis, group_size=group_size)


def random_qbits_tensor(shape, qtype, dtype, group_size, device):
    bits = qtype.bits
    qmax = 2**bits
    out_features, in_features = shape
    n_scales = out_features * in_features // group_size
    data_shape = (n_scales, group_size)
    data = torch.randint(0, qmax, data_shape, dtype=torch.uint8, device=device)
    scale_shape = (n_scales, 1)
    scale = torch.rand(scale_shape, dtype=dtype, device=device) / qmax
    shift_shape = (n_scales, 1)
    shift = torch.rand(shift_shape, dtype=dtype, device=device)
    return QBitsTensor(
        qtype, axis=0, group_size=group_size, size=shape, stride=(in_features, 1), data=data, scale=scale, shift=shift
    )


def assert_similar(a, b, atol=None, rtol=None):
    """Verify that the cosine similarity of the two inputs is close to 1.0 everywhere"""
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    if atol is None:
        # We use torch finfo resolution
        atol = torch.finfo(a.dtype).resolution
    if rtol is None:
        # Please refer to that discussion for default rtol values based on the float type:
        # https://scicomp.stackexchange.com/questions/43111/float-equality-tolerance-for-single-and-half-precision
        rtol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1e-1}[a.dtype]
    sim = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    if not torch.allclose(sim, torch.tensor(1.0, dtype=sim.dtype), atol=atol, rtol=rtol):
        max_deviation = torch.min(sim)
        raise ValueError(f"Alignment {max_deviation:.8f} deviates too much from 1.0 with atol={atol}, rtol={rtol}")


def get_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()
    elif device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory()
    return None


_run_staging = os.getenviron("HUGGINGFACE_CO_STAGING", False)


def is_staging_test(test_case):
    """
    Decorator marking a test as a staging test.

    Those tests will run using the staging environment of huggingface.co instead of the real model hub.
    """
    if not _run_staging:
        return unittest.skip("test is staging test")(test_case)
    else:
        return pytest.mark.is_staging_test()(test_case)
