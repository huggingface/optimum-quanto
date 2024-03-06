#include <torch/extension.h>

torch::Tensor mm_2bit(torch::Tensor &input, torch::Tensor &other, torch::Tensor& other_scale);
