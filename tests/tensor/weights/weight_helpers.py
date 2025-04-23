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
from helpers import assert_similar, random_tensor


def check_weight_qtensor_linear(qweight, batch_size, tokens, use_bias, rel_max_err=0.0):
    dtype = qweight.dtype
    device = qweight.device
    out_features, in_features = qweight.shape
    inputs = torch.rand((batch_size, tokens, in_features), dtype=dtype, device=device)
    bias = random_tensor((out_features,), dtype=dtype, device=device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    # Verify global alignment
    assert_similar(out, qout)
    # Also look for outliers
    mean_val = out.abs().max()
    max_err = (out - qout).abs().max()
    rel_max_err = max_err / mean_val
    # These values were evaluated empirically without any optimized kernels.
    rtol = {"cpu": 1e-2, "cuda": 2e-2, "mps": 1e-2, "xpu": 2e-2}[device.type]
    assert rel_max_err < rtol, (
        f"Maximum error {max_err:.2f} is too high for input of mean value {mean_val:.2f} ({rel_max_err * 100:.2f} %)"
    )
