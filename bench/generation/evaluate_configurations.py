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
import json

import torch
from evaluate_model import evaluate
from gen_barchart import gen_barchart
from transformers import AutoConfig

from optimum.quanto import qtype


def evaluate_model_configurations(
    model_id: str, metric: str, device: torch.device, batch_size: int = 32, dtype: torch.dtype = torch.float16
):
    weights = [
        "int4",
        "int8",
        "float8",
    ]

    activations = [
        "none",
        "float8",
    ]

    def short_name(qtype: qtype):
        return {
            "none": "f16" if dtype == torch.float16 else "bf16",
            "int4": "i4",
            "int8": "i8",
            "float8": "f8",
        }[qtype]

    results = {}

    # Evaluate float16/bfloat16 model
    config_name = f"W{short_name('none')}A{short_name('none')}"
    print(f"{model_id}[{config_name}]:")
    results[config_name] = evaluate(model_id, metric, "quanto", "none", "none", batch_size, device, dtype)
    # Evaluate quantized models
    for w in weights:
        for a in activations:
            config_name = f"W{short_name(w)}A{short_name(a)}"
            print(f"{model_id}[{config_name}]:")
            results[config_name] = evaluate(model_id, metric, "quanto", w, a, batch_size, device, dtype)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized model predictions on Lambada Dataset")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="The name of the trained Model.",
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
    parser.add_argument("--metric", type=str, default="prediction", choices=["latency", "prediction", "perplexity"])
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size during evaluation.")
    parser.add_argument("--dtype", type=str, help="Use the following dtype to load the model.")
    parser.add_argument("--json", action="store_true", help="Dump the results to a json file.")
    parser.add_argument("--png", action="store_true", help="Generate a PNG.")
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

    if args.dtype is None:
        config = AutoConfig.from_pretrained(args.model)
        dtype = getattr(config, "torch_dtype", torch.float16)
    else:
        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    results = evaluate_model_configurations(args.model, args.metric, device, batch_size=args.batch_size, dtype=dtype)
    if args.json:
        model_name = args.model.split("/")[-1]
        json_path = f"{model_name}-{args.metric}.json"
        with open(json_path, "w") as fp:
            json.dump({model_name: results}, fp, indent=4)
    if args.png:
        if args.metric == "latency":
            title = f"{args.model}: Mean latency per token"
            label = "Latency (ms)"
        elif args.metric == "prediction":
            title = f"{args.model}: Prediction accuracy on Lambada dataset"
            label = "Accuracy"
        elif args.metric == "perplexity":
            title = f"{args.model}: Perplexity evaluated on WikiText dataset"
            label = "Perplexity"
        gen_barchart(args.model, title, label, results, dtype)


if __name__ == "__main__":
    main()
