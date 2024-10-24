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


def unpack_int32_to_uint8(packed: torch.Tensor, bits: int):
    """Unpack a packed int32 tensor to a larger uint8 tensor

    Args:
        packed (`torch.Tensor`):
            The packed integer tensor
        bits: (`int`):
            The number of bits of each packed value.

    Returns:
        An unpacked uint8 `torch.Tensor` expanded along the last dimension.
    """
    total_bits = 32
    shifts = torch.arange(0, total_bits, bits, device=packed.device)

    # Unpack column-wise
    unpacked = torch.bitwise_right_shift(packed[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    unpacked = unpacked.reshape(unpacked.shape[0], -1)

    # Convert to unsigned
    unpacked = torch.bitwise_and(unpacked, (2**bits) - 1)

    return unpacked.to(torch.uint8)
