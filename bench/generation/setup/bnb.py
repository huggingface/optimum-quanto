# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def setup(
    model_id: str,
    weights: str,
    activations: str,
    device: torch.device,
):
    if activations != "none":
        raise ValueError("Activation quantization is not supported by BitsAndBytes")
    if weights == "int4":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4")
    elif weights == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError("BitsAndBytes only supports int4 and int8 weights.")
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    quantization_config.bnb_4bit_compute_dtype = dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, quantization_config=quantization_config
    )

    return model, tokenizer
