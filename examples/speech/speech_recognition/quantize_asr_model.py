# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# REQUIRES: librosa, soundfile
import argparse
import io
import time
from functools import partial

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from optimum.quanto import Calibration, freeze, qint4, qint8, quantize


def map_to_feats(batch, processor):
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["input_features"] = input_features
    batch["reference"] = processor.tokenizer.normalize(batch["text"])

    return batch


def transcribe_batch(batch, model, processor):
    with torch.no_grad():
        features = torch.from_numpy(np.array(batch["input_features"], dtype=np.float32)).squeeze(1)
        predicted_ids = model.generate(features.to(model.device))
    transcription = [processor.decode(ids) for ids in predicted_ids]
    batch["prediction"] = [processor.tokenizer.normalize(x) for x in transcription]
    return batch


def evaluate_model(model, processor, dataset, metric: evaluate.EvaluationModule, batch_size=10):
    map_fn = partial(transcribe_batch, model=model, processor=processor)
    start = time.time()
    result = dataset.map(map_fn, batched=True, batch_size=batch_size)
    end = time.time()
    score = 100 * metric.compute(references=result["reference"], predictions=result["prediction"])
    print(score)
    print(f"{len(result)} sentences evaluated in {end - start:.2f} s. {metric.name} = {score}")


def keyword_to_itype(k):
    return {"none": None, "int8": qint8, "int4": qint4}[k]


def main():
    parser = argparse.ArgumentParser(description="Transformers Whisper Example")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-medium",
        help="The name of the trained Model.",
    )
    parser.add_argument(
        "--samples", type=int, default=872, help="The number of librispeech samples to use for evaluation."
    )
    parser.add_argument("--batch_size", type=int, default=10, help="The batch size to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8"])
    parser.add_argument("--device", type=str, default=None, help="The device to use for evaluation.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("USING CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print("USING CPU")
    else:
        device = torch.device(args.device)

    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.config.forced_decoder_ids = None
    processor = WhisperProcessor.from_pretrained(args.model)
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    processed_dataset = dataset.map(lambda x: map_to_feats(x, processor))
    wer = load("wer")

    print("Float model:")
    evaluate_model(model, processor, processed_dataset, wer, args.batch_size)
    weights = keyword_to_itype(args.weights)
    activations = keyword_to_itype(args.activations)
    quantize(model, weights=weights, activations=activations)
    if activations is not None:
        print("Calibrating ...")
        with Calibration():
            evaluate_model(model, processor, processed_dataset, wer, args.batch_size)
    freeze(model)
    print(f"Quantized model (w: {args.weights}, a: {args.activations})")
    evaluate_model(model, processor, processed_dataset, wer, args.batch_size)
    b = io.BytesIO()
    torch.save(model.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b)
    model_reloaded = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    quantize(model_reloaded, weights=weights, activations=activations)
    model_reloaded.load_state_dict(state_dict)
    print("Serialized quantized model")
    evaluate_model(model, processor, processed_dataset, wer, args.batch_size)


if __name__ == "__main__":
    main()
