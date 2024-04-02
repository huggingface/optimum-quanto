import functools
import gc

import pytest
import torch
from packaging import version

from quanto import absmax_scale, qint8, quantize_activation, quantize_weight


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


def random_tensor(shape, dtype=torch.float32):
    # Return a random tensor between -1. and 1.
    return torch.rand(shape, dtype=dtype) * 2 - 1


def random_qactivation(shape, qtype=qint8, dtype=torch.float32):
    t = random_tensor(shape, dtype)
    scale = absmax_scale(t, qtype=qtype)
    return quantize_activation(t, qtype=qtype, scale=scale)


def random_qweight(shape, qtype, dtype=torch.float32, axis=0, group_size=None):
    t = random_tensor(shape, dtype)
    return quantize_weight(t, qtype=qtype, axis=axis, group_size=group_size)


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
