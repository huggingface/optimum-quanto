import argparse
import os
import time
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import AutoModel

from quanto.quantization import QTensor, freeze, quantize
from quanto.quantization.calibrate import calibration


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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=250, metavar="N", help="input batch size for testing (default: 250)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--model", type=str, default="dacorvo/mnist-mlp", help="The name of the trained Model.")
    parser.add_argument("--per_axis", action="store_true", help="Quantize activations per-axis.")
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
    quantize(model)
    print("Quantized model (dynamic weights only)")
    test(model, device, test_loader)
    print("Quantized model (dynamic weights and activations)")
    with calibration(per_axis=args.per_axis):
        test(model, device, test_loader)
    print("Tuning quantized model for one epoch")
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
    train(50, model, device, train_loader, optimizer, 1)
    print("Quantized tuned model (dynamic weights and static activations)")
    test(model, device, test_loader)
    print("Quantized frozen model (static weights and activations)")
    freeze(model)
    test(model, device, test_loader)
    with TemporaryDirectory() as tmpdir:
        mlp_file = os.path.join(tmpdir, "mlp.pt")
        torch.save(model.state_dict(), mlp_file)
        model_reloaded = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        quantize(model_reloaded)
        model_reloaded.load_state_dict(torch.load(mlp_file), assign=True)
    print("Serialized quantized model (static weights and activations)")
    test(model_reloaded, device, test_loader)


if __name__ == "__main__":
    main()
