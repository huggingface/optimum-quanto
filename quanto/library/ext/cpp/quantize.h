#include <torch/extension.h>

torch::Tensor quantize_symmetric(const torch::Tensor& input,
                                 const torch::Tensor& scale,
                                 at::ScalarType dtype);
