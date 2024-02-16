import argparse
import time
from contextlib import nullcontext

import numpy as np
import torch
from tqdm.auto import tqdm

from quanto.library import disable_extensions


def get_quantize_symmetric_bench(src_dtype, dst_dtype, per_axis, device):
    a = torch.rand([10240, 10240], dtype=src_dtype).to(device)
    scale = torch.fill((10240,), 0.5) if per_axis else torch.tensor(0.5)
    scale = scale.to(src_dtype).to(device)

    def bench_fn():
        return torch.ops.quanto.quantize_symmetric(a, scale, dst_dtype)

    return bench_fn


def get_unpack_bench(bits, device):
    qmax = 2**bits
    a = torch.randint(0, qmax, [10240, 10240], dtype=torch.uint8).to(device)

    def bench_fn():
        return torch.ops.quanto.unpack(a, bits)

    return bench_fn


def timing(get_bench_func, device, iterations=10):
    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        else:
            torch.cpu.synchronize()

    def timing_event(device):
        if device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        elif device.type == "mps":
            return torch.mps.Event(enable_timing=True)

        class CPUEvent:
            def __init__(self):
                self.time = None

            def record(self):
                self.time = time.time()

            def elapsed_time(self, other):
                assert self.time is not None
                assert other.time is not None
                return (other.time - self.time) * 1000

        return CPUEvent()

    synchronize(device)

    bench_func = get_bench_func(device)
    # Warmup to load library
    bench_func()
    latencies = np.empty((iterations, 2))
    for i in tqdm(range(iterations)):
        for j, context in enumerate([disable_extensions(), nullcontext()]):
            start_event = timing_event(device)
            end_event = timing_event(device)
            synchronize(device)
            start_event.record()
            with context:
                bench_func()
            end_event.record()
            synchronize(device)
            latencies[i, j] = start_event.elapsed_time(end_event)
    return np.mean(latencies[:, 0]), np.mean(latencies[:, 1])


GET_BENCH_FUNCTIONS = {
    "quantize_symmetric_fp32_int8_per_tensor": lambda device: get_quantize_symmetric_bench(
        torch.float32, torch.int8, False, device
    ),
    "unpack_2bit": lambda device: get_unpack_bench(2, device),
    "unpack_4bit": lambda device: get_unpack_bench(4, device),
}


def main():
    parser = argparse.ArgumentParser(description="Kernel benchmark")
    parser.add_argument("--kernel", type=str, default=None, help="The kernel to benchmark. None to test all of them")
    parser.add_argument("--device", type=str, default=None, help="The device to use for benchmark.")
    parser.add_argument("--it", type=int, default=10, help="The number of benchmark iterations")
    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    all_kernels = GET_BENCH_FUNCTIONS.keys()
    kernels = all_kernels if args.kernel is None else [args.kernel]
    for kernel in kernels:
        get_bench_fn = GET_BENCH_FUNCTIONS[kernel]
        python_ms, ext_ms = timing(get_bench_fn, device, iterations=args.it)
        ratio = python_ms / ext_ms
        print(f"\n{kernel}[{device.type}]: python = {python_ms:.3f} ms, ext = {ext_ms:.3f} ms, ratio = {ratio:.1f}x")


if __name__ == "__main__":
    main()
