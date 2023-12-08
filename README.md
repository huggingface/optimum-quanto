# Quanto

**DISCLAIMER**: this package is still an early prototype (pre-beta version), and not (yet) an HuggingFace product. Expect breaking changes and drastic modifications in scope and features.

🤗 Quanto is a python quantization toolkit that provides several features that are either not supported or limited by the base [pytorch quantization tools](https://pytorch.org/docs/stable/quantization.html):

- all features are available in eager mode (works with non-traceable models),
- quantized models can be placed on any device (including CUDA),
- automatically inserts quantization and dequantization stubs,
- automatically inserts quantized functional operations,
- automatically inserts quantized modules (see below the list of supported modules),
- provides a seamless workflow from float model to dynamic to static quantized model,
- supports quantized model serialization as a `state_dict`,
- uses integer matrix multiplications (`mm`) on CUDA devices.

Features yet to be implemented:

- quantize clone (quantization happens in-place for now),
- dynamic activations smoothing,
- integer batched matrix multiplications (`bmm`) on CUDA devices,
- integer matrix multiplications for CPU and MPS devices,
- quantized operators fusion (`mm` followed by dequantization is the most common use case),
- support `int4` weights,
- compatibility with [torch compiler](https://pytorch.org/docs/stable/torch.compiler.html) (aka dynamo).

## Quantized modules

Thanks to a seamless propagation mechanism through quantized tensors, only a few modules working as quantized
tensors insertion points are actually required.

The following modules can be quantized:

- [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (QLinear).
Weights are quantized to `int8`, and biases are not quantized. Inputs and outputs can be quantized to `int8`.
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
Weights and biases are __not__ quantized. Outputs can be quantized to `int8`.

The next modules to be implemented are:

- Conv2D.

## Limitations and design choices

Quanto uses a strict affine quantization scheme (no zero-point).

Quanto does not support mixed-precision quantization.

Quanto dynamically quantizes weights until a model is frozen: this slows
down inference a bit, but is required if the model needs to be tuned.

Activations are only quantized if the model has been calibrated to evaluate the activation scales.

Although `Quanto` uses integer activations and weights, the current implementation falls
back to `float32` operations for integer inputs if there is no support for the corresponding integer
operation on the target device (which means pretty much all operations except 2D matrix multiplications on CUDA devices).

Note: integer operations cannot be performed in `float16` as a fallback because this format is very bad at representing
`integer` and will likely lead to overflows in intermediate calculations.

## Performances

**DISCLAIMER**: these are preliminary observations gathered from a panel of models, and not an actual performance report.

In terms of accuracy:

- models using only quantized int8 weights do not seem to suffer any drop in accuracy,
- models using also quantized activations do suffer from moderate to severe accuracy drops.

In terms of speed:

- models using quantized weights only are very slightly slower than the original float model due to the weight dequantization,
- models using also quantized activations are significantly faster on CUDA devices,
- models using also quantized activations are significantly slower on CPU and MPS devices, where fallbacks are triggered.

The weight storage and on-device memory usage should:

- be equivalent for a model with dynamic weights,
- lower for a model with static weights.

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
