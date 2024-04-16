#include <torch/extension.h>

torch::Tensor awq_v2_gemm_f16i4(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scales, torch::Tensor _zeros);
