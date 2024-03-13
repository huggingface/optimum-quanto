#include <torch/extension.h>

torch::Tensor unpack(torch::Tensor &t, int bits, torch::IntArrayRef orig_shape, int axis);