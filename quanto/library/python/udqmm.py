import torch


@torch.library.impl("quanto_py::udqmm", "default")
def udqmm(
    input: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
    zeropoint: torch.Tensor,
    axis: int,
    bits: int,
    orig_shape: torch.Size,
    unpacked_shape: torch.Size,
) -> torch.Tensor:
    # we transpose it back, so it is simpler to unpack since we have the pack + transposed weights
    weights = weights.transpose(0, 1)
    unpacked_weights = torch.ops.quanto.unpack(weights, bits)
    # TODO : we should proably add that in unpack with with arg unpacked_shape.
    # Depends if the weights have been transposed or not
    # if not transposed, we need to do unpacked_weights[: unpacked_shape[0]]
    unpacked_weights = unpacked_weights[: unpacked_shape[1]]
    # transpose back
    unpacked_weights = unpacked_weights.transpose(0, 1)
    shifted_weights = unpacked_weights.to(torch.int8) - zeropoint
    scaled_weights = shifted_weights.to(scale.dtype) * scale
    ungrouped_weights = torch.ops.quanto.ungroup(scaled_weights, axis, orig_shape)
    return torch.ops.aten.mm(input, ungrouped_weights)


import torch
from types import SimpleNamespace as ns
import math



def blk_kernel2d(f, blocks, threads, *args):
    """
    This simulates the parallel processing on GPUs
    """
    for i0 in range(blocks.y):
        for i1 in range(blocks.x):
            for j0 in range(threads.y):
                for j1 in range(threads.x): 
                    f(ns(x=i1,y=i0), ns(x=j1,y=j0), threads, *args)


def matmul_bk(blockidx, threadidx, blockdim, m, n, out, h, w, k):
    """
    This is the kernel code for Matrix Multiplication. 
    
    It is implemented in Python, but will be changed to MPS.
    """
    r = blockidx.y*blockdim.y + threadidx.y
    c = blockidx.x*blockdim.x + threadidx.x
    
    if (r>=h or c>=w): return
    o = 0.0

    for i in range(k): o += m[r*k+i] * n[i*w+c]
    
    out[r*w+c] = o


@torch.library.impl("quanto_py::combined_unpack_mm_kernel", "default")
def combined_unpack_mm_kernel(input: torch.Tensor, weights: torch.Tensor, scale: torch.Tensor, bits: int):
    """
    This is driver code for the kernel

    """
    unpacked = []
    values_per_item = 8 // bits

    def rshift(t: torch.Tensor, bits: int):
        if t.device.type == "mps":
            # rshift is not supported on MPS device
            return t // (2**bits)
        return t >> bits

    # Unpack each set of values independently
    for i in range(values_per_item):
        mask = 2 ** (bits * (i + 1)) - 1
        unpacked.append(rshift(weights & mask, bits * i))
    

    unpacked_weights = torch.cat(unpacked).to(torch.uint8)

    scaled_weights = unpacked_weights * scale

    # TODO: future: pass `orig_shape`
    if scaled_weights.size(0) != input.size(1):
        last_dim = int(scaled_weights.numel() / input.size(1))
        original_shape = (input.size(1), last_dim)
        scaled_weights = scaled_weights.reshape(original_shape)

    h, k = input.shape
    k2, w = scaled_weights.shape
    assert k == k2, f"Size Mismatch: {k=}, {k2=}"
    output = torch.zeros(h, w, dtype=input.dtype)
    threads_per_block = ns(x=16,y=16)
    blocks = ns(x=math.ceil(w/threads_per_block.x), y=math.ceil(h/threads_per_block.y))
    blk_kernel2d(matmul_bk, blocks, threads_per_block,
                 input.flatten(), scaled_weights.flatten(), output.flatten(), h, w, k)

    return output
