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

import argparse
import time
from tempfile import NamedTemporaryFile

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from torchvision import datasets, transforms
from transformers import AutoConfig, AutoModel

from optimum.quanto import (
    Calibration,
    QTensor,
    freeze,
    qfloat8,
    qint4,
    qint8,
    quantization_map,
    quantize,
    requantize,
)


def test(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        start = time.time()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, QTensor):
                output = output.dequantize()
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        end = time.time()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set evaluated in {:.2f} s: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            end - start, test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, QTensor):
            output = output.dequantize()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def keyword_to_itype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=250, metavar="N", help="input batch size for testing (default: 250)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--model", type=str, default="dacorvo/mnist-mlp", help="The name of the trained Model.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8"])
    parser.add_argument("--device", type=str, default=None, help="The device to use for evaluation.")
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

    dataset_kwargs = {"batch_size": args.batch_size}
    if torch.cuda.is_available():
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        dataset_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **dataset_kwargs)
    dataset2 = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **dataset_kwargs)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    print("Float model")
    test(model, device, test_loader)
    weights = keyword_to_itype(args.weights)
    activations = keyword_to_itype(args.activations)
    quantize(model, weights=weights, activations=activations)
    if activations is not None:
        print("Calibrating ...")
        with Calibration():
            test(model, device, test_loader)
    print(f"Quantized model (w: {args.weights}, a: {args.activations})")
    test(model, device, test_loader)
    print("Tuning quantized model for one epoch")
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
    train(50, model, device, train_loader, optimizer, 1)
    print("Quantized tuned model")
    test(model, device, test_loader)
    print("Quantized frozen model")
    freeze(model)
    test(model, device, test_loader)
    # Serialize model to a state_dict, save it to disk and reload it
    with NamedTemporaryFile() as tmp_file:
        save_file(model.state_dict(), tmp_file.name)
        state_dict = load_file(tmp_file.name)
    model_reloaded = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    # Create an empty model
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    with init_empty_weights():
        model_reloaded = AutoModel.from_config(config, trust_remote_code=True)
    # Requantize it using the serialized state_dict
    requantize(model_reloaded, state_dict, quantization_map(model), device)
    print("Serialized quantized model")
    test(model_reloaded, device, test_loader)


if __name__ == "__main__":
    main()
