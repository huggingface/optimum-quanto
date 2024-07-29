import argparse
import gc

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize


def detect(model, processor, image, texts):
    inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Note: boxes need to be visualized on the padded, unnormalized image
    # hence we'll set the target image sizes (height, width) based on that
    def get_preprocessed_image(pixel_values):
        pixel_values = pixel_values.squeeze().cpu().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[
            :, None, None
        ]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    unnormalized_image = get_preprocessed_image(inputs.pixel_values)

    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    if len(boxes) == 0:
        print("None of the specified labels were detected")
        return

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


def get_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()
    elif device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory()
    return None


def keyword_to_qtype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/owlv2-base-patch16")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--texts", type=str, nargs="+", required=True)
    parser.add_argument("--weights", type=str, default="none", choices=["none", "int4", "int8", "float8"])
    parser.add_argument("--exclude-heads", action="store_true", help="Do not quantize detection heads")
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # MPS backend does not support torch.float64 that is required for owl models
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(args.model, low_cpu_mem_usage=True).to(device)

    weights_qtype = keyword_to_qtype(args.weights)
    if weights_qtype is not None:
        if args.exclude_heads:
            quantize(model.owlv2, weights=weights_qtype)
        else:
            quantize(model, weights=weights_qtype)
        freeze(model)

    memory = get_device_memory(device)
    if memory is not None:
        memory_gb = memory / 2**30
        print(f"{device.type} device memory: {memory_gb:.2f} GB.")

    image_path = args.image
    if image_path.startswith("http"):
        image_path = requests.get(args.image, stream=True).raw
    image = Image.open(image_path)

    texts = [args.texts]
    detect(model, processor, image, texts)


if __name__ == "__main__":
    main()
