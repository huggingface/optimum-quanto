#include "mm.h"
#include <torch/extension.h>

torch::Tensor dqmm(torch::Tensor &input, torch::Tensor &other, torch::Tensor& other_scale) {
    return torch::mm(input, other * other_scale);
}
