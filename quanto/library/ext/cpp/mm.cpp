#include "mm.h"
#include <torch/extension.h>

torch::Tensor dqmm(torch::Tensor &input, torch::Tensor &other, torch::Tensor& other_scale) {
    if (other.is_floating_point()) {
        // An explicit type promotion avoids errors with float8 tensors (but is slower for integer)
        auto pother = other.to(other_scale.scalar_type());
        return torch::mm(input, pother * other_scale);
    }
    return torch::mm(input, other * other_scale);
}
