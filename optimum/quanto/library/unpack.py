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


torch.library.define("quanto::unpack", "(Tensor self, int bits) -> Tensor")


@torch.library.impl("quanto::unpack", "default")
def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Un-Pack int4 / int2 weights (packed in a uint8) into a torch.uint8 tensor
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
    return torch.cat(unpacked).to(torch.uint8)
