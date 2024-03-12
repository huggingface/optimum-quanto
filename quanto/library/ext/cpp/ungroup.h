#include <torch/extension.h>

torch::Tensor ungroup(torch::Tensor &grouped, int axis, torch::IntArrayRef orig_shape);