# Quanto

**DISCLAIMER**: this package is still an early prototype (pre-beta version), and not (yet) an HuggingFace product. Expect breaking changes and drastic modifications in scope and features.

🤗 Quanto is a python quantization toolkit that provides several features that are either not supported or limited by the base [pytorch quantization tools](https://pytorch.org/docs/stable/quantization.html):

- all features are available in eager mode (works with non-traceable models),
- quantized models can be placed on any device (including CUDA),
- automatically inserts quantization and dequantization stubs,
- automatically inserts quantized functional operations,
- automatically inserts quantized modules (see below the list of supported modules),
- provides a seamless workflow from float model to dynamic to static quantized model,
- supports quantized model serialization as a `state_dict`.

Features yet to be implemented:

- quantize clone (quantization happens in-place for now),
- smart calibration to decide if activations must be per-tensor or per-axis,
- optimized integer kernels,
- quantized operators fusion,
- support `int4` weights,
- compatibility with [torch compiler](https://pytorch.org/docs/stable/torch.compiler.html) (aka dynamo).

## Supported modules

The following modules can be quantized:

- [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (QLinear).
Weights are quantized to `int8`, and biases to `int32`. Outputs are quantized to `int8`.
If the activations are quantized per-axis, the weights are also quantized per-axis.
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
Weights and biases are __not__ quantized. Outputs are quantized to `int8`.

The next modules to be implemented are:

- LLamaRMSNorm.

## Limitations and design choices

Quanto uses a strict affine quantization scheme (no zero-point).

Quanto does not support mixed-precision quantization.

Quanto dynamically quantizes weights until a model is frozen: this slows
down inference a lot, but is required if the model needs to be tuned.

Although `Quanto` uses integer activations and weights, the current implementation falls
back to `float32` operations for integer inputs, and some quantization operations are
redundant, which means that inference is slower, even for frozen models.
The weight storage and on-device memory usage should however be lower.

## Installation

Quanto is available as a pip package.

```
pip install quanto
```

## Quantization workflow

Quanto does not make a clear distinction between dynamic and static quantization: models are always dynamically quantized,
but their weights can later be "frozen" to integer values.

A typical quantization workflow would consist in the following steps:

1. Quantize

The first step converts a standard float model into a dynamically quantized model.

```
quantize(model)
```

2. Calibrate (optional)

Activations are quantized using a default `[-1, 1]` range which can lead to severe clipping and/or inaccurate values.

Quanto supports a calibration mode that allows to adjust the activation ranges while passing representative samples through the quantized model.

```
with calibration():
    model(samples)
```
Note that during calibration, all activations and weights are dequantized and inference is evaluated with float precision.

3. Tune, aka Quantization-Aware-Training (optional)

If the performances of the model are too degraded, one can tune it for a few epochs to recover the float model performances.

```
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data).dequantize()
    loss = torch.nn.functional.nll_loss(output, target)
    loss.backward()
    optimizer.step()
```

4. Freeze integer weights

When freezing a model, its float weights are replaced by quantized integer weights.

```
freeze(model)
```

Please refer to the [examples](https://github.com/huggingface/quanto/tree/main/examples) for instantiations of that workflow.

## Per-axis versus per-tensor

By default, all weights and activations are quantized per-tensor, as it is the quantization scheme that is compatible with the
highest number of operations with minimal changes.

This can however lead to serious quantization errors if the corresponding tensors contain large outlier values: typically, this will
lead to quantized tensors with most values set to zero (except the outliers).

To work around that issue, activations can also be quantized per-axis:

```
with calibration(per_axis=True):
    model(samples)
```

This is unlikely to produce a fully quantized graph however, since a lot of operations require per-tensor inputs, leading to a dequantization
down the line.

Typically, in a transformer model, per-axis activations of Q, K, V linear projections will be dequantized when they are split by heads, and the
downstream matmul will be performed on float tensors.

**In other words, quantizing activations per-axis will most of the time be equivalent to a weight-only quantization.**

## Implementation details

Under the hood, Quanto uses a `torch.Tensor` subclass (`QTensor`) to dispatch `aten` base operations to integer operations.

All integer operations accept `QTensor` with `int8` data.

Most arithmetic operations return a `QTensor` with `int32` data.

In addition to the quantized tensors, Quanto uses quantized modules as substitutes to some base torch modules to:

- store quantized weights,
- gather input and output scales to rescale QTensor `int32` data to `int8`.

Eventually, the produced quantized graph should be passed to a specific inductor backend to fuse rescale into the previous operation.

Examples of fused operations can be found in https://github.com/Guangxuan-Xiao/torch-int.
