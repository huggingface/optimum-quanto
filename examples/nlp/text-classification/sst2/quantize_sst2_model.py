import argparse
import os
import time
from tempfile import TemporaryDirectory

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from quanto.quantization import freeze, quantize
from quanto.quantization.calibrate import calibration


def evaluate_model(model, tokenizer, dataset, device, batch_size):
    p = pipeline("sentiment-analysis", model, tokenizer=tokenizer, device=device)
    results = p(KeyDataset(dataset, "sentence"), batch_size=batch_size)
    start = time.time()
    pred_labels = [0 if result["label"] == "NEGATIVE" else 1 for result in results]
    end = time.time()
    accuracy = np.sum(np.equal(pred_labels, dataset["label"])) / len(pred_labels)
    print(f"{len(pred_labels)} sentences evaluated in {end - start:.2f} s. accuracy = {accuracy}")


def main():
    parser = argparse.ArgumentParser(description="Transformers SST2 Example")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="The name of the trained Model.",
    )
    parser.add_argument("--samples", type=int, default=872, help="The number of sst2 samples to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size to use for evaluation.")
    parser.add_argument("--per_axis", action="store_true", help="Quantize activations per-axis.")
    parser.add_argument("--device", type=str, default=None, help="The device to use for evaluation.")
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

    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("sst2", split=f"validation[:{args.samples}]")

    print("Float model")
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    quantize(model)
    print("Quantized model (dynamic weights only)")
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    print("Quantized model (dynamic weights and activations)")
    with calibration(per_axis=args.per_axis):
        evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    freeze(model)
    print("Quantized model (static weights and activations)")
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    with TemporaryDirectory() as tmpdir:
        model_file = os.path.join(tmpdir, "model.pt")
        torch.save(model.state_dict(), model_file)
        model_reloaded = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
        quantize(model_reloaded)
        model_reloaded.load_state_dict(torch.load(model_file), assign=True)
    print("Serialized quantized model (static weights and activations)")
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)


if __name__ == "__main__":
    main()
