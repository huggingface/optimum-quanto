#include "udqmm.h"
#include "unpack.h"

#include <iostream>
#include <torch/extension.h>

using namespace std;

torch::Tensor udqmm(torch::Tensor &input, torch::Tensor &weights, torch::Tensor& scale, torch::Tensor& zeropoint, int axis, int bits, torch::IntArrayRef orig_shape) {
    torch::Tensor unpacked_weights = unpack(weights, bits);
    torch::Tensor dq_output = (unpacked_weights.to(torch::kInt8) - zeropoint.to(torch::kInt8)) * scale;

    torch::Tensor ungrouped_output;

    // Ungroup TODO : put on its own function
    if (dq_output.sizes() == orig_shape){
        ungrouped_output = dq_output;
    }
    if (axis == 0) {
        ungrouped_output = torch::reshape(dq_output, orig_shape);
    }
    // Finish axis = 1 case

    return torch::mm(input, ungrouped_output);
}