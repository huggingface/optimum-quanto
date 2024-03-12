#include "udqmm.h"
#include "unpack.h"
#include "ungroup.h"

#include <iostream>
#include <torch/extension.h>

using namespace std;

torch::Tensor udqmm(torch::Tensor &input, torch::Tensor &weights, torch::Tensor &scale, torch::Tensor &zeropoint, int axis, int bits, torch::IntArrayRef orig_shape, torch::IntArrayRef unpacked_shape) {
    TORCH_CHECK(zeropoint.scalar_type() == torch::kInt8, "zeropoint must have scalar type: torch.int8");
    // transpose the weights
    weights = weights.transpose(0, 1);
    torch::Tensor unpacked_weights = unpack(weights, bits);
    // slice along the first dim from index 0 to unpacked_shape[1] since we transposed the weights
    unpacked_weights = unpacked_weights.slice(0, 0, unpacked_shape[1]);
    // transpose back since we finished unpacking the weights
    unpacked_weights = unpacked_weights.transpose(0, 1);

    torch::Tensor dq_output = (unpacked_weights.to(torch::kInt8) - zeropoint).to(scale.dtype()) * scale;

    torch::Tensor ungrouped_output = ungroup(dq_output, axis, orig_shape);

    return torch::mm(input, ungrouped_output);
}