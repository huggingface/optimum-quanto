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

    def get_int_matmul(device):
        if device.type == ("cuda"):
            return torch._int_mm
        return torch.matmul

    def avg_time(f, it):
        return timeit.Timer(f).timeit(it) / it

    int_matmul = get_int_matmul(device)

    # Resstrictions for accelerated integer matmul:
    # - input matrices must be 2D
    # - the collapsing dimension must be a multiple of 8
    A = torch.randint(1, 10, [2400, 3200]).type(torch.int8).to(device)
    B = torch.randint(1, 10, [3200, 4800]).type(torch.int8).to(device)

    print(f"Evaluating integer matmul on {device.type}:")
    # Warmup (slow)
    int_matmul(A, B)
    # Average on several calls
    t = avg_time(lambda: int_matmul(A, B), args.it) * 1000
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
