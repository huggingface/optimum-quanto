#include <torch/extension.h>


static torch::Tensor unpack_bytes_4bit(torch::Tensor &W_q) {
	return torch::cat({
                      (W_q & 0x0F),
                      (W_q & 0xF0).__rshift__(4)
                    },
                    0);
}

static torch::Tensor unpack_bytes_2bit(torch::Tensor &W_q) {
	return torch::cat({
                      (W_q & 0x03),
                      (W_q & 0x0C).__rshift__(2),
                      (W_q & 0x30).__rshift__(4),
                      (W_q & 0xC0).__rshift__(6)
                    },
                    0);
}

torch::Tensor unpack_bytes(torch::Tensor &W_q, int bits) {
    switch(bits) {
      case 4:
        return unpack_bytes_4bit(W_q);
      case 2:
        return unpack_bytes_2bit(W_q);
      default:
        throw std::invalid_argument("Can only unpack 2-bit or 4-bit tensors.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unpack", &unpack_bytes, "unpack");
}
