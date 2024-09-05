#include <torch/extension.h>

torch::Tensor marlin_gemm(torch::Tensor &a, torch::Tensor &b_q_weight,
                          torch::Tensor &b_scales, torch::Tensor &workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k);
