import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quanto import Calibration, freeze, qfloat8, qint4, qint8, qtype, quantize


@torch.no_grad()
def _evaluate_model(model, tokenizer, dataset, device, batch_size, samples=None, log=True):
    model.eval()
    # The task is to predict the last token of the input.
    total, hit = 0, 0
    start = time.time()
    for batch in dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        labels = input_ids[:, -1]
        # Pass only the first tokens
        outputs = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
        preds = outputs.logits[:, -1, :].argmax(dim=-1)
        total += labels.size(0)
        hit += (preds == labels).sum().item()
        if samples is not None and total >= samples:
            break
    end = time.time()
    acc = hit / total
    if log:
        print(f"{total} sequences evaluated in {end - start:.2f} s. accuracy = {acc:.2f}")
    return acc


def prediction_accuracy(
    model_id: str,
    weights: qtype,
    activations: qtype,
    device: torch.device,
    samples: int = None,
    batch_size: int = 32,
    validation_batch: int = 4,
    seed: int = 1,
):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    test_dataset, cal_dataset = load_dataset("lambada", split=["test", "validation"])
    if weights is not None or activations is not None:
        quantize(model, weights=weights, activations=activations)
        if activations is not None:
            print("Calibrating ...")
            cal_dataset.shuffle(seed)
            with Calibration():
                cal_samples = batch_size * validation_batch
                _evaluate_model(model, tokenizer, cal_dataset, device, batch_size, samples=cal_samples, log=False)
        freeze(model)
        print("Evaluating model predictions ...")
    return _evaluate_model(model, tokenizer, test_dataset, device, batch_size, samples=samples)


def keyword_to_qtype(k):
    return {
        "none": None,
        "int4": qint4,
        "int8": qint8,
        "float8": qfloat8,
    }[k]


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized model predictions on Lambada Dataset")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="The name of the trained Model.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="The number of samples to use for evaluation (defaults to the full test set).",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="The batch_size for evaluation (and calibration).")
    parser.add_argument("--validation_batch", type=int, default=4, help="The number of batch to use for calibration.")
    parser.add_argument(
        "--weights",
        type=str,
        default="int8",
        choices=["none", "int4", "int8", "float8"],
    )
    parser.add_argument(
        "--activations",
        type=str,
        default="int8",
        choices=["none", "int8", "float8"],
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
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

    weights = keyword_to_qtype(args.weights)
    activations = keyword_to_qtype(args.activations)
    print(f"{args.model} (w: {args.weights}, a: {args.activations})")
    prediction_accuracy(
        args.model,
        weights,
        activations,
        device,
        samples=args.samples,
        batch_size=args.batch_size,
        validation_batch=args.validation_batch,
    )


if __name__ == "__main__":
    main()
