import argparse
import os
import random
import time
from tempfile import TemporaryDirectory

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
        max_new_tokens=100,
        attention_mask=inputs.attention_mask.to(device),
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )
    end = time.time()
    generated_code = tokenizer.decode(outputs[0])
    print(f"Code generation took {end - start:.2f} s.")
    print(generated_code)


@torch.no_grad()
def calibrate_model(model, tokenizer, dataset, device, batch_size):
    model.eval()
    for batch in dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        # Just predict one token to calibrate the model
        model(input_ids=input_ids, attention_mask=attention_mask)


def main():
    parser = argparse.ArgumentParser(description="Transformers OPT Example")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/codegen-350M-mono",
        help="The name of the trained Model.",
    )
    parser.add_argument("--samples", type=int, default=100, help="The number of samples to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch_size for evaluation (and calibration).")
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

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    dataset = load_dataset("openai_humaneval", split=f"test[:{args.samples}]")

    sample_id = random.randint(0, args.samples - 1)
    prompt = dataset["prompt"][sample_id]
    print("Original code prompt")
    print(prompt)
    print("Float model")
    generate(model, tokenizer, device, prompt)
    # Quantize model
    quantize(model)
    # Test inference
    print("Quantized model")
    generate(model, tokenizer, device, prompt)
    # Test inference with calibration
    print("Quantized calibrated model")
    with calibration():
        calibrate_model(model, tokenizer, dataset, device, args.batch_size)
    # Freeze model
    freeze(model)
    print("Quantized frozen model")
    generate(model, tokenizer, device, prompt)
    # Now save the model and reload it to verify quantized weights are restored
    with TemporaryDirectory() as tmpdir:
        opt_file = os.path.join(tmpdir, "opt.pt")
        torch.save(model.state_dict(), opt_file)
        # Reinstantiate a model with float weights
        model_reloaded = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        quantize(model_reloaded)
        # When reloading we must assign instead of copying to force quantized tensors assignment
        model_reloaded.load_state_dict(torch.load(opt_file), assign=True)
    print("Quantized model with serialized integer weights")
    generate(model, tokenizer, device, prompt)


if __name__ == "__main__":
    main()
