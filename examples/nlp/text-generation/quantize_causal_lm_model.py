import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quanto import Calibration, freeze, qfloat8_e4m3fn, qfloat8_e5m2, qint8, quantize


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
def evaluate_model(model, tokenizer, dataset, device, batch_size, samples=None, log=True):
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


def keyword_to_itype(k):
    return {"none": None, "int8": qint8, "fp8_e5m2": qfloat8_e5m2, "fp8_e4m3": qfloat8_e4m3fn}[k]


def main():
    parser = argparse.ArgumentParser(description="Transformers Causal LM Example")
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
    parser.add_argument("--weights", type=str, default="int8", choices=["int8"], help="Only int8 is supported.")
    parser.add_argument(
        "--activations",
        type=str,
        default="int8",
        choices=["none", "int8", "fp8_e5m2", "fp8_e4m3"],
        help="One of none, int8, fp8_e5m2, fp8_e4m3.",
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
    parser.add_argument(
        "--no-streamline",
        action="store_false",
        help="Do not remove consecutive quantize/dequantize (not recommended).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Provide detailed feedback on the console during calibration."
    )
    parser.add_argument("--skip_float", action="store_true", help="Do not run comparison with float model.")
    parser.add_argument("--skip_generation", action="store_true", help="Do not generate outputs.")
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

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    test_dataset, cal_dataset = load_dataset("lambada", split=["test", "validation"])

    prompt = "One of my fondest memory is"
    if not args.skip_float:
        print(f"{args.model} ({model.config.torch_dtype})")
        if not args.skip_generation:
            generate(model, tokenizer, device, prompt)
        evaluate_model(model, tokenizer, test_dataset, device, args.batch_size, samples=args.samples)
    weights = keyword_to_itype(args.weights)
    activations = keyword_to_itype(args.activations)
    quantize(model, weights=weights, activations=activations)
    if activations is not None:
        print("Calibrating ...")
        cal_dataset.shuffle(args.seed)
        with Calibration(streamline=args.no_streamline, debug=args.debug):
            cal_samples = args.batch_size * args.validation_batch
            evaluate_model(model, tokenizer, cal_dataset, device, args.batch_size, samples=cal_samples, log=False)
    freeze(model)
    print(f"{args.model} (w: {args.weights}, a: {args.activations})")
    if not args.skip_generation:
        generate(model, tokenizer, device, prompt)
    evaluate_model(model, tokenizer, test_dataset, device, args.batch_size, samples=args.samples)


if __name__ == "__main__":
    main()
