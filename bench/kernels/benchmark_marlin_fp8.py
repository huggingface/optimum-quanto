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
import argparse
from typing import Optional

import numpy as np
import torch

from optimum.quanto.tensor.weights.marlin.packed import pack_fp8_as_int32


M_SHAPES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_SHAPES = [4096]
K_SHAPES = [4096]


def run_benchmark(
    m: Optional[int],
    n: Optional[int],
    k: Optional[int],
    n_runs: int,
    n_warmup: int,
    dtype: torch.dtype = torch.float16,
):
    print(f"\n----------- m={m}, n={n}, k={k}")
    n_tokens = m
    in_features = k
    out_features = n

    assert m is not None

    device = torch.device("cuda")
    inputs = torch.rand(n_tokens, in_features, dtype=dtype, device=device)

    other_shape = (in_features, out_features)
    other_data = torch.rand(other_shape, dtype=dtype, device=device).to(torch.float8_e4m3fn)
    other_data_int32 = pack_fp8_as_int32(other_data)
    perm = torch.empty(0, dtype=torch.int, device=device)

    other_data_repack = torch.ops.quanto.gptq_marlin_repack(
        b_q_weight=other_data_int32, perm=perm, size_k=in_features, size_n=out_features, num_bits=8
    )
    other_scale = torch.rand(1, dtype=dtype, device=device)
    other_scale = other_scale.repeat(1, out_features)

    workspace = torch.zeros(out_features // 64 * 16, dtype=torch.int, device=device)

    latencies_marlin_fp8 = []
    latencies_torch = []
    with torch.no_grad():
        for i in range(n_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_event.record()

            _ = torch.ops.quanto.fp8_marlin_gemm(
                a=inputs,
                b_q_weight=other_data_repack,
                b_scales=other_scale,
                workspace=workspace,
                num_bits=8,
                size_m=n_tokens,
                size_n=out_features,
                size_k=in_features,
            )
            end_event.record()
            torch.cuda.synchronize(device)

            latency_ms = start_event.elapsed_time(end_event)
            if i >= n_warmup:
                latencies_marlin_fp8.append(latency_ms)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_event.record()
            other = other_data.to(dtype) * other_scale
            _ = torch.matmul(inputs, other)
            end_event.record()
            torch.cuda.synchronize(device)

            latency_ms = start_event.elapsed_time(end_event)
            if i >= n_warmup:
                latencies_torch.append(latency_ms)

    mean_latency_torch = np.mean(latencies_torch)
    mean_latency_marlin_fp8 = np.mean(latencies_marlin_fp8)
    print("mean_latency_torch:", mean_latency_torch)
    print("mean_latency_marlin_fp8:", mean_latency_marlin_fp8)

    return mean_latency_torch, mean_latency_marlin_fp8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marlin FP8 kernel benchmark")
    parser.add_argument("--nruns", type=int, default=20, help="The number of benchmark iterations")
    parser.add_argument("--nwarmup", type=int, default=2, help="The number of warmup iterations (deducted from nruns)")
    parser.add_argument(
        "--m",
        type=int,
        help="m dimension of A=m*k",
        default=None,
    )
    parser.add_argument(
        "--n",
        type=int,
        help="n dimension of B=k*n (out_features)",
        default=None,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="k dimension of A=m*k and B=k*n (in_features), hidden_size",
        default=None,
    )
    args = parser.parse_args()

    if args.m is not None:

        def shape_generator():
            yield (args.m, args.n, args.k)

    else:

        def shape_generator():
            for m in M_SHAPES:
                for n in N_SHAPES:
                    for k in K_SHAPES:
                        yield (m, n, k)

    result = "m,n_out,k_in,torch_latency_ms,marlin_fp8_latency_ms\n"
    for m, n, k in shape_generator():
        mean_latency_torch, mean_latency_marlin_fp8 = run_benchmark(m, n, k, args.nruns, args.nwarmup)

        result += (
            ",".join(
                [
                    str(m),
                    str(n),
                    str(k),
                    f"{mean_latency_torch:.4f}",
                    f"{mean_latency_marlin_fp8:.4f}",
                ]
            )
            + "\n"
        )

    print("\nResults:")
    print(result)
