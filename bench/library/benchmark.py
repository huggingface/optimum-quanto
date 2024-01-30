import argparse
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from quanto.tensor.core import int2, int4, unpack_weights


def get_unpack_bench(bits, device):
    qmax = 2**bits
    a = torch.randint(0, qmax, [10240, 10240], dtype=torch.uint8).to(device)
    bitsdtype = int2 if bits == 2 else int4

    def torch_fn():
        return unpack_weights(a, bitsdtype)

    def kernel_fn():
        return torch.ops.quanto.unpack(a, bits)

    return [torch_fn, kernel_fn]


def timing(get_bench_functions, device, iterations=10):
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

    latencies = np.empty((iterations, 2))
    for i in tqdm(range(iterations)):
        bench_functions = get_bench_functions(device)
        for j, fn in enumerate(bench_functions):
            start_event = timing_event(device)
            end_event = timing_event(device)
            synchronize(device)
            start_event.record()
            fn()
            end_event.record()
            synchronize(device)
            latencies[i, j] = start_event.elapsed_time(end_event)
    return np.mean(latencies[:, 0]), np.mean(latencies[:, 1])


GET_BENCH_FUNCTIONS = {
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
    all_kernels = ["unpack_2bit", "unpack_4bit"]
    kernels = all_kernels if args.kernel is None else [args.kernel]
    for kernel in kernels:
        get_bench_functions = GET_BENCH_FUNCTIONS[kernel]
        torch_ms, kernel_ms = timing(get_bench_functions, device, iterations=args.it)
        ratio = torch_ms / kernel_ms
        print(
            f"\n{kernel}[{device.type}]: torch = {torch_ms:.3f} ms, kernel = {kernel_ms:.3f} ms, ratio = {ratio:.1f}x"
        )


if __name__ == "__main__":
    main()
