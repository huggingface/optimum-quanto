#include <torch/extension.h>

torch::Tensor udqmm(torch::Tensor &input, torch::Tensor &weights, torch::Tensor& scale, torch::Tensor& zeropoint, int axis, int bits, torch::IntArrayRef orig_shape);
