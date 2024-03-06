#include "mm_2bit.h"
#include "unpack.h"

#include <torch/extension.h>

torch::Tensor mm_2bit(torch::Tensor &input, torch::Tensor &weights, torch::Tensor& scale) {
    torch::Tensor unpacked_weights = unpack(weights, 2);
    return torch::mm(input, unpacked_weights * scale);
}