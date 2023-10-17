import argparse
import os
import time
from tempfile import TemporaryDirectory

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from quanto.quantization import freeze, quantize
from quanto.quantization.calibrate import calibration


def evaluate_model(model, tokenizer, dataset, device):
    p = pipeline("sentiment-analysis", model, tokenizer=tokenizer, device=device)
    results = p(KeyDataset(dataset, "sentence"))
    start = time.time()
    pred_labels = [0 if result["label"] == "NEGATIVE" else 1 for result in results]
    end = time.time()
    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(predictions=pred_labels, references=dataset["label"])["accuracy"]
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
    dataset = load_dataset("sst2", split="validation")

    print("Float model")
    evaluate_model(model, tokenizer, dataset, device)
    # Quantize model
    quantize(model)
    # Test inference
    print("Quantized model")
    evaluate_model(model, tokenizer, dataset, device)
    # Test inference with calibration
    print("Quantized calibrated model")
    with calibration():
        evaluate_model(model, tokenizer, dataset, device)
    # Freeze model
    freeze(model)
    print("Quantized frozen model")
    evaluate_model(model, tokenizer, dataset, device)
    # Now save the model and reload it to verify quantized weights are restored
    with TemporaryDirectory() as tmpdir:
        gpt2_file = os.path.join(tmpdir, "gpt2.pt")
        torch.save(model.state_dict(), gpt2_file)
        # Reinstantiate a model with float weights
        model_reloaded = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
        quantize(model_reloaded)
        # When reloading we must assign instead of copying to force quantized tensors assignment
        model_reloaded.load_state_dict(torch.load(gpt2_file), assign=True)
    print("Quantized model with serialized integer weights")
    evaluate_model(model_reloaded, tokenizer, dataset, device)


if __name__ == "__main__":
    main()
