// Copyright 2024 The HuggingFace Team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "unpack.h"
#include <torch/extension.h>


static torch::Tensor unpack_4bit(torch::Tensor &t) {
	return torch::cat({
                      (t & 0x0F),
                      (t & 0xF0).__rshift__(4)
                    },
                    0);
}

static torch::Tensor unpack_2bit(torch::Tensor &t) {
	return torch::cat({
                      (t & 0x03),
                      (t & 0x0C).__rshift__(2),
                      (t & 0x30).__rshift__(4),
                      (t & 0xC0).__rshift__(6)
                    },
                    0);
}

torch::Tensor unpack(torch::Tensor &t, int bits) {
    TORCH_CHECK(t.scalar_type() == torch::kUInt8, "Unsupported data type: ", t.scalar_type());
    switch(bits) {
      case 4:
        return unpack_4bit(t);
      case 2:
        return unpack_2bit(t);
      default:
        throw std::invalid_argument("Can only unpack 2-bit or 4-bit tensors.");
    }
}
