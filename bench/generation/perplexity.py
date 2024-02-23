import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quanto import Calibration, freeze, qfloat8, qint4, qint8, qtype, quantize


@torch.no_grad()
def _calibrate(model, tokenizer, dataset, device, batch_size, num_batches):
    i = 1
    for batch in dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        model(input_ids, attention_mask=attention_mask)
        i += 1
        if i > num_batches:
            break


@torch.no_grad()
def _perplexity(model, tokenizer, dataset, device, stride=512, seq_len=None):
    max_length = model.config.max_position_embeddings
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    if seq_len is None:
        seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()


def perplexity(
    model_id: str,
    weights: qtype,
    activations: qtype,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 1,
    stride: int = 512,
):
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    test_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if weights is not None:
        print("Quantizing")
        quantize(model, weights=weights, activations=activations)
        if activations is not None:
            print("Calibrating")
            with Calibration():
                _calibrate(model, tokenizer, test_set, device, batch_size=batch_size, num_batches=4)
        print("Freezing")
        freeze(model)

    print("Evaluating perplexity")
    return _perplexity(model, tokenizer, test_set, device, stride)


def keyword_to_qtype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]


def main():
    parser = argparse.ArgumentParser(description="Generate bechmark")
    parser.add_argument(
        "--model", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B", help="The model to use for benchmark"
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for benchmark.")
    parser.add_argument("--stride", type=int, default=512, help="The stride to use when evaluating perplexity.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch_size for evaluation (and calibration).")
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
    args = parser.parse_args()
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
    result = perplexity(args.model, weights, activations, device, batch_size=args.batch_size, stride=args.stride)
    print(result)


if __name__ == "__main__":
    main()
