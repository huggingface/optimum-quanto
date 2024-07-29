#include <torch/library.h>
#include <torch/all.h>
#include <stdint.h>

#ifndef _gptq_marlin_repack_cuh
#define _gptq_marlin_repack_cuh

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

#endif
