import argparse
import json

import torch
from gen_barchart import gen_barchart
from latency import latency
from perplexity import perplexity
from prediction import prediction_accuracy

from quanto import qfloat8, qint4, qint8, qtype


def evaluate_model_configurations(
    model_id: str, metric: str, device: torch.device, batch_size: int = 32, seed: int = 1
):
    weights = [
        qint4,
        qint8,
        qfloat8,
    ]

    activations = [
        None,
        qint8,
        qfloat8,
    ]

    def short_name(qtype: qtype):
        return {
            None: "f16",
            qint4: "i4",
            qint8: "i8",
            qfloat8: "f8",
        }[qtype]

    results = {}

    def get_results(model_id: str, w: qtype, a: qtype, device: torch.device, seed: int = 1):
        if metric == "latency":
            return latency(model_id, w, a, device, batch_size=batch_size, seed=seed)
        elif metric == "prediction":
            return prediction_accuracy(model_id, w, a, device, batch_size=batch_size, seed=seed)
        elif metric == "perplexity":
            return perplexity(model_id, w, a, device, batch_size=batch_size, seed=seed)

    # Evaluate float16 model
    print(f"{model_id}[Wf16Af16]:")
    results["Wf16Af16"] = get_results(model_id, None, None, device, seed=seed)
    # Evaluate quantized models
    for w in weights:
        for a in activations:
            config_name = f"W{short_name(w)}A{short_name(a)}"
            print(f"{model_id}[{config_name}]:")
            results[config_name] = get_results(model_id, w, a, device, seed=seed)

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

    results = evaluate_model_configurations(
        args.model, args.metric, device, batch_size=args.batch_size, seed=args.seed
    )
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
        gen_barchart(args.model, title, label, results)


if __name__ == "__main__":
    main()
