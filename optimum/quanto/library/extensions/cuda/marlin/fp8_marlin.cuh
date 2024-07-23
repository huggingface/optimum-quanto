// #pragma once
#include <torch/all.h>
#include <stdint.h>


// #ifndef _fp8_marlin_cuh
// #define _fp8_marlin_cuh

// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// assert(0);
// #else
torch::Tensor fp8_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                              torch::Tensor& b_scales, torch::Tensor& workspace,
                              int64_t num_bits, int64_t size_m, int64_t size_n,
                              int64_t size_k);
// #endif

// #endif