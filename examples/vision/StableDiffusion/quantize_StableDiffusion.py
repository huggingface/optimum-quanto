import argparse
from quanto import quantize, freeze, qint4, qint8, qfloat8_e4m3fn
import torch
import torch.utils.benchmark as benchmark
from diffusers import DiffusionPipeline

CKPT = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 50
WARM_UP_ITERS = 5
PROMPT = "ghibli style, a fantasy landscape with castles"

TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
UNET_DTYPES = {"fp8": qfloat8_e4m3fn, "int8": qint8, "int4": qint4}

def load_pipeline(torch_dtype, unet_dtype=None):
    pipe = DiffusionPipeline.from_pretrained(
        CKPT, torch_dtype=torch_dtype, use_safetensors=True
    ).to("cuda")

    if unet_dtype:
        quantize(pipe.unet, weights=unet_dtype)
        freeze(pipe.unet)

    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_inference(pipe, batch_size=1):
    _ = pipe(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_images_per_prompt=batch_size,
        generator=torch.manual_seed(0),
    )


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    return f"{(t0.blocked_autorange().mean):.3f}"


def bytes_to_giga_bytes(bytes):
    return f"{(bytes / 1024 / 1024 / 1024):.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--torch_dtype", type=str, default="fp32", choices=list(TORCH_DTYPES.keys()))
    parser.add_argument("--unet_dtype", type=str, default=None, choices=list(UNET_DTYPES.keys()))
    args = parser.parse_args()

    pipeline = load_pipeline(
        TORCH_DTYPES[args.torch_dtype], UNET_DTYPES[args.unet_dtype] if args.unet_dtype else None
    )

    for _ in range(WARM_UP_ITERS):
        run_inference(pipeline, args.batch_size)

    time = benchmark_fn(run_inference, pipeline, args.batch_size)
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
    print(
        f"batch_size: {args.batch_size}, torch_dtype: {args.torch_dtype}, unet_dtype: {args.unet_dtype}  in {time} seconds."
    )
    print(f"Memory: {memory}GB.")

    img_name = f"bs@{args.batch_size}-dtype@{args.torch_dtype}-unet_dtype@{args.unet_dtype}.png"
    image = pipeline(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_images_per_prompt=args.batch_size,
    ).images[0]
    image.save(img_name)