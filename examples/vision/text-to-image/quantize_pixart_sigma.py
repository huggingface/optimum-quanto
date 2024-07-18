import argparse
import gc

import torch
from diffusers import DiffusionPipeline

from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize


NUM_INFERENCE_STEPS = 50

TORCH_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
QTYPES = {
    "fp8": qfloat8,
    "int8": qint8,
    "int4": qint4,
    "none": None,
}


def load_pipeline(model_id, torch_dtype, qtype=None, device="cpu"):
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True).to(device)

    if qtype:
        quantize(pipe.transformer, weights=qtype)
        freeze(pipe.transformer)
        quantize(pipe.text_encoder, weights=qtype)
        freeze(pipe.text_encoder)

    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()
    elif device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--prompt", type=str, default="ghibli style, a fantasy landscape with castles")
    parser.add_argument("--torch_dtype", type=str, default="fp16", choices=list(TORCH_DTYPES.keys()))
    parser.add_argument("--qtype", type=str, default=None, choices=list(QTYPES.keys()))
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
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

    pipeline = load_pipeline(
        args.model_id, TORCH_DTYPES[args.torch_dtype], QTYPES[args.qtype] if args.qtype else None, device
    )

    print(f"torch_dtype: {args.torch_dtype}, qtype: {args.qtype}.")
    memory = get_device_memory(device)
    if memory is not None:
        memory_gb = memory / 2**30
        print(f"{device.type} device memory: {memory_gb:.2f} GB.")

    if args.qtype == "int4" and device.type == "CUDA":
        raise ValueError("This example does not work (yet) for int4 on CUDA")

    img_name = f"pixart-sigma-dtype@{args.torch_dtype}-qtype@{args.qtype}.png"
    image = pipeline(
        prompt=args.prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_images_per_prompt=1,
        generator=torch.manual_seed(0),
    ).images[0]
    image.save(img_name)
