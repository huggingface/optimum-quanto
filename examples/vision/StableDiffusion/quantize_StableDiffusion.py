import argparse
import gc

import torch
import torch.utils.benchmark as benchmark
from diffusers import DiffusionPipeline

from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize


CKPT = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 50
WARM_UP_ITERS = 5
PROMPT = "ghibli style, a fantasy landscape with castles"

TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
UNET_QTYPES = {
    "fp8": qfloat8,
    "int8": qint8,
    "int4": qint4,
    "none": None,
}


def load_pipeline(torch_dtype, unet_dtype=None, device="cpu"):
    pipe = DiffusionPipeline.from_pretrained(CKPT, torch_dtype=torch_dtype, use_safetensors=True).to(device)

    if unet_dtype:
        quantize(pipe.unet, weights=unet_dtype)
        freeze(pipe.unet)

    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_inference(pipe, batch_size=1):
    _ = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
        generator=torch.manual_seed(0),
    )


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    return f"{(t0.blocked_autorange().mean):.3f}"


def bytes_to_giga_bytes(bytes):
    return f"{(bytes / 1024 / 1024 / 1024):.3f}"


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
    parser.add_argument("--prompt", type=str, default="ghibli style, a fantasy landscape with castles")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--torch_dtype", type=str, default="fp32", choices=list(TORCH_DTYPES.keys()))
    parser.add_argument("--unet_qtype", type=str, default=None, choices=list(UNET_QTYPES.keys()))
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
        TORCH_DTYPES[args.torch_dtype], UNET_QTYPES[args.unet_qtype] if args.unet_qtype else None, device
    )

    for _ in range(WARM_UP_ITERS):
        run_inference(pipeline, args.batch_size)

    time = benchmark_fn(run_inference, pipeline, args.batch_size)
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
    get_device_memory(device)
    print(
        f"batch_size: {args.batch_size}, torch_dtype: {args.torch_dtype}, unet_dtype: {args.unet_qtype}  in {time} seconds."
    )
    print(f"Memory: {memory}GB.")

    img_name = f"bs@{args.batch_size}-dtype@{args.torch_dtype}-unet_dtype@{args.unet_qtype}.png"
    image = pipeline(
        prompt=args.prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_images_per_prompt=args.batch_size,
    ).images[0]
    image.save(img_name)
