#include <torch/extension.h>

torch::Tensor unpack(const torch::Tensor &input, int bits, torch::IntArrayRef orig_shape, int axis);
