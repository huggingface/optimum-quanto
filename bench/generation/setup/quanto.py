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

import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize


@torch.no_grad()
def calibrate(model, tokenizer, batch_size, batches):
    samples = batch_size * batches
    cal_dataset = load_dataset("lambada", split=["validation"])[0]
    model.eval()
    total = 0
    for batch in cal_dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        model(input_ids, attention_mask=attention_mask)
        total += input_ids.size(0)
        if total >= samples:
            break


def setup(
    model_id: str,
    weights: str,
    activations: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    weights = keyword_to_qtype(weights)
    activations = keyword_to_qtype(activations)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    if weights is not None or activations is not None:
        print("Quantizing")
        start = time.time()
        quantization_root = model
        if hasattr(model, "model"):
            quantization_root = model.model
        quantize(quantization_root, weights=weights, activations=activations)
        if activations is not None:
            print("Calibrating")
            with Calibration():
                calibrate(model, tokenizer, batch_size, batches=4)
        print("Freezing")
        freeze(model)
        print(f"Finished: {time.time() - start:.2f}")
    return model, tokenizer


def keyword_to_qtype(k):
    return {
        "none": None,
        "int4": qint4,
        "int8": qint8,
        "float8": qfloat8,
    }[k]
