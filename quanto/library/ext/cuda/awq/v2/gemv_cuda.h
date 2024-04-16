#pragma once
#include <torch/extension.h>

torch::Tensor awq_v2_gemv_f16i4(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int m,
    int n,
    int k,
    int group_size);
