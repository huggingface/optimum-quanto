#include "unpack.h"
#include <torch/extension.h>


static torch::Tensor unpack_4bit(torch::Tensor &t, int axis) {
	return torch::cat({
                      (t & 0x0F),
                      (t & 0xF0).__rshift__(4)
                    },
                    axis);
}

static torch::Tensor unpack_2bit(torch::Tensor &t, int axis) {
	return torch::cat({
                      (t & 0x03),
                      (t & 0x0C).__rshift__(2),
                      (t & 0x30).__rshift__(4),
                      (t & 0xC0).__rshift__(6)
                    },
                    axis);
}

static torch::Tensor slice_along_axis(torch::Tensor& t, torch::IntArrayRef orig_shape, int axis) {
    return t.slice(axis, 0, orig_shape[axis]);
}

torch::Tensor unpack(torch::Tensor &t, int bits, torch::IntArrayRef orig_shape, int axis) {
    TORCH_CHECK(t.scalar_type() == torch::kUInt8, "Unsupported data type: ", t.scalar_type());
    switch(bits) {
      case 4: {
        auto output = unpack_4bit(t, axis);
        return slice_along_axis(output, orig_shape, axis);
      }
      case 2: {
        auto output = unpack_2bit(t, axis);
        return slice_along_axis(output, orig_shape, axis);
      }
      default:
        throw std::invalid_argument("Can only unpack 2-bit or 4-bit tensors.");
    }
}
