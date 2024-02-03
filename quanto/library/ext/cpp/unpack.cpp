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
    switch(bits) {
      case 4:
        return unpack_4bit(t);
      case 2:
        return unpack_2bit(t);
      default:
        throw std::invalid_argument("Can only unpack 2-bit or 4-bit tensors.");
    }
}
