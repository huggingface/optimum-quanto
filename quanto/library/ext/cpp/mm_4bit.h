#include <torch/extension.h>

torch::Tensor mm_4bit(torch::Tensor &input, torch::Tensor &other, torch::Tensor& other_scale);
