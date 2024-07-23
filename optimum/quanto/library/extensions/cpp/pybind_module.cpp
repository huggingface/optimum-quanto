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
#include "unpack.h"

// !IMPORTANT! Some python objects such as dtype, device, are not mapped to C++ types,
// and need to be explicitly converted using dedicated helpers before calling a C++ method.
// As a consequence, when an operation takes such an object as parameter, instead
// of creating a binding directly to the C++ method, you must create a binding to a
// lambda method that converts the unmapped types and calls the C++ method.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unpack", &unpack, "unpack");
}
