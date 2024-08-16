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
from packaging import version


def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    assert w.dim() == 2
    w = w.transpose(0, 1).contiguous()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert torch.isnan(scales).sum() == 0

    zeros = min_val + scales * (2 ** (n_bit - 1))
    assert torch.isnan(zeros).sum() == 0

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert torch.isnan(out).sum() == 0

    out = out.to(dtype=torch.int32).reshape(w.shape)

    # Scales and zeros for the same q-group should be contiguous, so we can
    # load as a 32-bit word
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

    return out, scales_and_zeros


def main():
    parser = argparse.ArgumentParser(description="Torch quantized int4 weight matmul benchmark")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="floating point type")
    parser.add_argument("--device", type=str, default=None, help="The device to use for the test.")
    parser.add_argument("--it", type=int, default=10, help="Number of iterations for average")
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

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    A = torch.rand([2400, 3200], dtype=dtype, device=device)
    B = torch.rand([3200, 4800], dtype=dtype, device=device)
    group_size = 128
    B_int32, B_scale_and_zeros = _group_quantize_tensor(B, n_bit=4, q_group_size=group_size)
    if version.parse(torch.__version__).release >= version.parse("2.5.0").release:
        B_uint8 = (B_int32[::, ::2] << 4 | B_int32[::, 1::2]).to(torch.uint8)
        B_packed = torch._convert_weight_to_int4pack(B_uint8, innerKTiles=2)
    else:
        B_packed = torch._convert_weight_to_int4pack(B_int32, innerKTiles=2)

    # Check quantized mm is close to float mm
    qout = torch._weight_int4pack_mm(A, B_packed, group_size, B_scale_and_zeros)
    out = torch.mm(A, B)

    mean_err = ((qout - out).abs() / out.abs()).mean()
    print(mean_err)

    print(f"Evaluating quantized int4 matmul on {device.type}:")
    # Warmup (slow)
    torch._weight_int4pack_mm(A, B_packed, group_size, B_scale_and_zeros)
    # Average on several calls
    t = avg_time(lambda: torch._weight_int4pack_mm(A, B_packed, group_size, B_scale_and_zeros), args.it) * 1000
    print(f"Average inference on {args.it} iterations: {t:.4f} ms")

    print(f"Evaluating {A.dtype} matmul on {device.type}:")

    # Warmup (slow)
    torch.mm(A, B)
    # Average on several calls
    t = avg_time(lambda: torch.mm(A, B), args.it) * 1000
    print(f"Average inference on {args.it} iterations: {t:.4f} ms")


if __name__ == "__main__":
    main()
