import argparse
import gc
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from quanto import freeze, quantize


@torch.no_grad()
def generate(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs.to(device)
    start = time.time()
    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=20,
        attention_mask=inputs.attention_mask,
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )
    end = time.time()
    generated_text = tokenizer.decode(outputs[0])
    print(f"Generated '{generated_text}' in [{end - start:.2f} s]")


def timing_cuda(model, tokenizer, device, batch_size=1, prompt_length=512, nb_tokens=512):
    generation_config = GenerationConfig(
        max_new_tokens=nb_tokens,
        min_new_tokens=nb_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1,
        do_sample=False,
        eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
    )
    model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.

    torch.cuda.synchronize()

    latencies = []
    input_ids = torch.randint(1, model.config.vocab_size - 1, size=(batch_size, prompt_length)).to(device)
    masks = torch.ones(batch_size, prompt_length, dtype=torch.int32).to(device)

    # mean over 10 batches
    for _ in tqdm(range(10)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        print(f"\nLatency per token: {latency_ms / generation_config.min_new_tokens:.3f} ms")
        latencies.append(latency_ms)

    return np.mean(latencies)


def main():
    parser = argparse.ArgumentParser(description="Generate bechmark")
    parser.add_argument("--quanto", action="store_true", help="Quantization using Quanto (W8A16)")
    parser.add_argument("--bnb_4bit", action="store_true", help="Quantization using bitandbytes 4bit")
    parser.add_argument("--bnb_8bit", action="store_true", help="Quantization using bitandbytes 8bit")
    args = parser.parse_args()
    device = 0
    quantization_config = None
    if args.bnb_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16
        )
    elif args.bnb_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    if quantization_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="cuda:0",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)

        if args.quanto:
            print("quantizing")
            start = time.time()
            quantize(model, weights=torch.int8, activations=None)
            freeze(model)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Finished: {time.time()-start}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    prompt = "One of my fondest memory is"
    generate(model, tokenizer, device, prompt)
    timing_cuda(model, tokenizer, device, batch_size=1, prompt_length=512, nb_tokens=512)


if __name__ == "__main__":
    main()
