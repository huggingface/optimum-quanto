#include "mm_4bit.h"
#include "unpack.h"

#include <torch/extension.h>

torch::Tensor mm_4bit(torch::Tensor &input, torch::Tensor &weights, torch::Tensor& scale) {
    torch::Tensor unpacked_weights = unpack(weights, 4);
    return torch::mm(input, unpacked_weights * scale);
}