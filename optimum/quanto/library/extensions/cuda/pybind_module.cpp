// Copyright 2024 The HuggingFace Team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
#include "awq/v2/gemm_cuda.h"
#include "awq/v2/gemv_cuda.h"
#include "unpack.h"
#include "marlin/fp8_marlin.cuh"
#include "marlin/gptq_marlin_repack.cuh"
#include <pybind11/pybind11.h>
#include <torch/library.h>

// !IMPORTANT! Some python objects such as dtype, device, are not mapped to C++ types,
// and need to be explicitly converted using dedicated helpers before calling a C++ method.
// As a consequence, when an operation takes such an object as parameter, instead
// of creating a binding directly to the C++ method, you must create a binding to a
// lambda method that converts the unmapped types and calls the C++ method.
// See the binding of quantize_symmetric for instance.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("awq_v2_gemm_f16i4", &awq_v2_gemm_f16i4, "awq_v2_gemm_f16i4");
  m.def("awq_v2_gemv_f16i4", &awq_v2_gemv_f16i4, "awq_v2_gemv_f16i4");
  m.def("unpack", &unpack, "unpack");
}

TORCH_LIBRARY(quanto_ext, m) {
  m.impl_abstract_pystub("optimum.quanto.library.ops");
  m.def("gptq_marlin_repack(Tensor b_q_weight, Tensor perm, int size_k, int size_n, int num_bits) -> Tensor");
  m.def("fp8_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, Tensor workspace, int num_bits, int size_m, int size_n, int size_k) -> Tensor");
}

TORCH_LIBRARY_IMPL(quanto_ext, CUDA, m) {
  m.impl("gptq_marlin_repack", &gptq_marlin_repack);
  m.impl("fp8_marlin_gemm", &fp8_marlin_gemm);
}