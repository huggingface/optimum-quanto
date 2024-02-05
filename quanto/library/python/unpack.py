import torch


@torch.library.impl("quanto_py::unpack", "default")
def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Un-Pack int4 / int2 weights (packed in a uint8) into a torch.int8 tensor
    What un-packing means? Assume we have packed 4 2-bit values in 8-bit
    (because torch does not have native support for 2-bit datatypes)

    > 1110 0100

    Unpacking them means retrieving the original 4 2-bit values:

    > 0000 0011 | 0000 0010 | 0000 0001 | 0000 0000

    Args:
        packed (`torch.Tensor`):
            The packed tensor in `torch.uint8` precision
        bits (`int`):
            The number of bits per encoded value. Can be 2 or 4.
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
        unpacked.append(rshift(packed & mask, bits * i))
    # Return the concatenated unpacked tensors
    return torch.cat(unpacked).to(torch.int8)
