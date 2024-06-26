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
import timeit

import torch


def main():
    parser = argparse.ArgumentParser(description="Torch integer matmul benchmark")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--device", type=str, default=None, help="The device to use for the test.")
    parser.add_argument("--it", type=int, default=100, help="Number of iterations for average")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    def avg_time(f, it):
        return timeit.Timer(f).timeit(it) / it

    # Resstrictions for accelerated integer matmul:
    # - input matrices must be 2D
    # - the collapsing dimension must be a multiple of 8
    A = torch.randint(1, 10, [2400, 3200]).type(torch.int8).to(device)
    B = torch.randint(1, 10, [3200, 4800]).type(torch.int8).to(device)

    print(f"Evaluating integer matmul on {device.type}:")
    # Warmup (slow)
    torch._int_mm(A, B)
    # Average on several calls
    t = avg_time(lambda: torch._int_mm(A, B), args.it) * 1000
    print(f"Average inference on {args.it} iterations: {t:.4f} ms")

    # Convert inputs to float

    def to_float(x):
        if x.device.type == ("cpu"):
            # matrix multiplication is not supported for float16 on CPU
            return x.to(torch.float32)
        return x.to(torch.float16)

    A = to_float(A)
    B = to_float(B)
    print(f"Evaluating {A.dtype} matmul on {device.type}:")

    # Warmup (slow)
    torch.matmul(A, B)
    # Average on several calls
    t = avg_time(lambda: torch.matmul(A, B), args.it) * 1000
    print(f"Average inference on {args.it} iterations: {t:.4f} ms")


if __name__ == "__main__":
    main()
