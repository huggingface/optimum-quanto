import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


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


def gen_barchart(model_id, title, label, results):
    activations = ("f16", "i8", "f8")
    weights = ("i4", "i8", "f8")
    series = {}
    reference = round(results["Wf16Af16"], 2)
    series["Weights f16"] = [reference,] * len(activations)
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
        save_path=f"{model_name}_{metric_name}.png",
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
