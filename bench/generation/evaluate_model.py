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
import importlib

import torch
from datasets import load_dataset
from metrics.latency import latency
from metrics.perplexity import perplexity
from metrics.prediction import prediction_accuracy


if importlib.util.find_spec("awq") is not None:
    from setup.awq import setup as awq_setup
if importlib.util.find_spec("bitsandbytes") is not None:
    from setup.bnb import setup as bnb_setup
if importlib.util.find_spec("hqq") is not None:
    from setup.hqq import setup as hqq_setup
from setup.quanto import setup as quanto_setup
from transformers import AutoConfig


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


def evaluate(
    model_id: str,
    metric: str,
    quantizer: str,
    weights: str,
    activations: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = None,
):
    if quantizer == "quanto":
        if dtype is None:
            config = AutoConfig.from_pretrained(model_id)
            dtype = getattr(config, "torch_dtype", torch.float16)
        model, tokenizer = quanto_setup(model_id, weights, activations, batch_size, device, dtype)
    elif quantizer == "awq":
        model, tokenizer = awq_setup(model_id, weights, activations, group_size=128)
    elif quantizer == "bnb":
        model, tokenizer = bnb_setup(model_id, weights, activations, device)
    elif quantizer == "hqq":
        model, tokenizer = hqq_setup(model_id, weights, activations, device)
    else:
        raise ValueError(f"Unsupported quantizer {quantizer}")
    dtype = next(model.parameters()).dtype
    weights = dtype if weights == "none" else weights
    activations = dtype if activations == "none" else activations
    print(f"Evaluating {model_id} {metric} with {weights} weights and {activations} activations.")
    if metric == "latency":
        return latency(model, tokenizer, device, batch_size=1, prompt_length=512, nb_tokens=512, iterations=3)
    elif metric == "prediction":
        return prediction_accuracy(model, tokenizer, batch_size)
    elif metric == "perplexity":
        return perplexity(model, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized model metrics")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="The name of the trained Model.",
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
    parser.add_argument("--metric", type=str, default="prediction", choices=["latency", "prediction", "perplexity"])
    parser.add_argument("--quantizer", type=str, default="quanto", choices=["quanto", "awq", "bnb", "hqq"])
    parser.add_argument(
        "--weights",
        type=str,
        default="none",
        choices=["none", "int4", "int8", "float8"],
    )
    parser.add_argument(
        "--activations",
        type=str,
        default="none",
        choices=["none", "int8", "float8"],
    )
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size during evaluation.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="none",
        choices=["none", "fp16", "bf16"],
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
    dtype = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    evaluate(args.model, args.metric, args.quantizer, args.weights, args.activations, args.batch_size, device, dtype)


if __name__ == "__main__":
    main()
