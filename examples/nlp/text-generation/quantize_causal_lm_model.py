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

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.quanto import Calibration, QuantizedModelForCausalLM, qfloat8, qint4, qint8


@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    start = time.time()
    outputs = model.generate(
        input_ids=inputs.input_ids.to(device),
        max_new_tokens=max_new_tokens,
        attention_mask=inputs.attention_mask.to(device),
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )
    end = time.time()
    generated_text = tokenizer.decode(outputs[0])
    print(f"Generated '{generated_text}' in [{end - start:.2f} s]")


@torch.no_grad()
def calibrate(model, tokenizer, dataset, device, batch_size, samples=None):
    model.eval()
    total = 0
    for batch in dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        model(input_ids, attention_mask=attention_mask)
        total += input_ids.size(0)
        if samples is not None and total >= samples:
            break


def keyword_to_itype(k):
    return {
        "none": None,
        "int4": qint4,
        "int8": qint8,
        "float8": qfloat8,
    }[k]


def main():
    parser = argparse.ArgumentParser(description="Transformers Causal LM Example")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="The name of the trained Model.",
    )
    parser.add_argument("--prompt", type=str, default="One of my fondest memory is", help="The generation prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="The maximum number of tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch_size for evaluation (and calibration).")
    parser.add_argument("--validation_batch", type=int, default=4, help="The number of batch to use for calibration.")
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Precision to load the initial model",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="int8",
        choices=["int4", "int8", "float8"],
    )
    parser.add_argument(
        "--activations",
        type=str,
        default="int8",
        choices=["none", "int8", "float8"],
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
    parser.add_argument(
        "--no-streamline",
        action="store_false",
        help="Do not remove consecutive quantize/dequantize (not recommended).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Provide detailed feedback on the console during calibration."
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch_dtype = (
        torch.float16
        if args.load_dtype == "float16"
        else torch.bfloat16
        if args.load_dtype == "bfloat16"
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    cal_dataset = load_dataset("lambada", split=["validation"])[0]

    print(f"{args.model} (w: {args.weights}, a: {args.activations})")
    weights = keyword_to_itype(args.weights)
    activations = keyword_to_itype(args.activations)
    qmodel = QuantizedModelForCausalLM.quantize(model, weights=weights, activations=activations)
    if activations is not None:
        print("Calibrating ...")
        cal_dataset.shuffle(args.seed)
        with Calibration(streamline=args.no_streamline, debug=args.debug):
            cal_samples = args.batch_size * args.validation_batch
            calibrate(qmodel, tokenizer, cal_dataset, device, args.batch_size, samples=cal_samples)
    generate(qmodel, tokenizer, device, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
