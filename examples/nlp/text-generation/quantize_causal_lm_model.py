import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quanto.quantization import calibration, freeze, quantize


@torch.no_grad()
def generate(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    start = time.time()
    outputs = model.generate(
        input_ids=inputs.input_ids.to(device),
        max_new_tokens=20,
        attention_mask=inputs.attention_mask.to(device),
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )
    end = time.time()
    generated_text = tokenizer.decode(outputs[0])
    print(f"Generated '{generated_text}' in [{end - start:.2f} s]")


@torch.no_grad()
def evaluate_model(model, tokenizer, dataset, device, batch_size):
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
    end = time.time()
    acc = hit / total
    print(f"{total} sequences evaluated in {end - start:.2f} s. accuracy = {acc:.2f}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Transformers Causal LM Example")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="The name of the trained Model.",
    )
    parser.add_argument("--samples", type=int, default=100, help="The number of samples to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch_size for evaluation (and calibration).")
    parser.add_argument("--per_axis", action="store_true", help="Quantize activations per-axis.")
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

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", low_cpu_mem_usage=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    dataset = load_dataset("lambada", split=f"validation[:{args.samples}]").shuffle()

    prompt = "One of my fondest memory is"
    print("Float model")
    generate(model, tokenizer, device, prompt)
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    quantize(model)
    print("Quantized model (dynamic weights only)")
    generate(model, tokenizer, device, prompt)
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    print("Quantized model (dynamic weights and activations)")
    with calibration(per_axis=args.per_axis):
        evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    freeze(model)
    print("Quantized model (static weights and activations)")
    generate(model, tokenizer, device, prompt)
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)


if __name__ == "__main__":
    main()
