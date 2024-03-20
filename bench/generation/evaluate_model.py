import argparse

import torch
from datasets import load_dataset
from metrics.latency import latency
from metrics.perplexity import perplexity
from metrics.prediction import prediction_accuracy

from setup.awq import setup as awq_setup
from setup.bnb import setup as bnb_setup
from setup.hqq import setup as hqq_setup
from setup.quanto import setup as quanto_setup


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
    model_id: str, metric: str, quantizer: str, weights: str, activations: str, batch_size: int, device: torch.device
):
    if quantizer == "quanto":
        model, tokenizer = quanto_setup(model_id, weights, activations, batch_size, device)
    elif quantizer == "awq":
        model, tokenizer = awq_setup(model_id, weights, activations)
    elif quantizer == "bnb":
        model, tokenizer = bnb_setup(model_id, weights, activations, device)
    elif quantizer == "hqq":
        model, tokenizer = hqq_setup(model_id, weights, activations, device)
    else:
        raise ValueError(f"Unsupported quantizer {quantizer}")
    if metric == "latency":
        return latency(model, tokenizer, device, batch_size=1, prompt_length=512, nb_tokens=512, iterations=5)
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
    evaluate(args.model, args.metric, args.quantizer, args.weights, args.activations, args.batch_size, device)


if __name__ == "__main__":
    main()
