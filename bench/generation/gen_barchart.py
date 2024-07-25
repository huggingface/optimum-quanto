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

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_bar_chart(title, labels, ylabel, series, save_path):
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")
    fig.set_figwidth(10)

    max_value = 0

    for attribute, measurement in series.items():
        max_value = max(max_value, max(measurement))
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=5)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width, labels)
    ax.legend(loc="upper left", ncols=4)
    ax.set_ylim(0, max_value * 1.2)

    plt.savefig(save_path)


def gen_barchart(model_id, title, label, results, dtype):
    dtype_str = "f16" if dtype is torch.float16 else "bf16"
    activations = (dtype_str, "f8")
    weights = ("i4", "i8", "f8")
    series = {}
    reference = round(results[f"W{dtype_str}A{dtype_str}"], 2)
    series[f"Weights {dtype_str}"] = [
        reference,
    ] * len(activations)
    for w in weights:
        name = f"Weights {w}"
        series[name] = []
        for a in activations:
            result = results[f"W{w}A{a}"]
            series[name].append(round(result, 2))
    model_name = model_id.replace("/", "-")
    metric_name = label.replace(" ", "_").replace("(", "_").replace(")", "_")
    save_bar_chart(
        title=title,
        labels=[f"Activations {a}" for a in activations],
        series=series,
        ylabel=label,
        save_path=f"{model_name}_{dtype_str}_{metric_name}.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str, help="A benchmark result file (.json).")
    parser.add_argument("--title", type=str, required=True, help="The graph title.")
    parser.add_argument("--label", type=str, required=True, help="The graph vertical label.")
    args = parser.parse_args()
    with open(args.benchmark) as f:
        benchmark = json.load(f)
        for model_id, results in benchmark.items():
            gen_barchart(model_id, args.title, args.label, results)


if __name__ == "__main__":
    main()
