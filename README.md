# Quanto

**DISCLAIMER**: This package is still an early prototype (pre-beta version), and not (yet) an HuggingFace product. Expect breaking changes and drastic modifications in scope and features.

🤗 Quanto is a python quantization toolkit that provides several features that are either not supported or limited by the base [pytorch quantization tools](https://pytorch.org/docs/stable/quantization.html):

- all features are available in eager mode (works with non-traceable models),
- quantized models can be placed on any device (including CUDA),
- automatically inserts quantization and dequantization stubs,
- automatically inserts quantized functional operations,
- automatically inserts quantized modules (see below the list of supported modules),
- provides a seamless workflow from a float model to a dynamic to a static quantized model,
- supports quantized model serialization as a `state_dict`,
- uses integer matrix multiplications (`mm`) on CUDA devices,
- supports float8 activations.

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

### Tensors

At the heart of quanto is a Tensor subclass that corresponds to:
- the projection of a source Tensor into the optimal range for a given destination type,
- the mapping of projected values to the destination type.

For floating-point destination types, the mapping is done by the native pytorch cast (i.e. `Tensor.to()`).

For integer destination types, the mapping is a simple rouding opeation (i.e. `torch.round()`).

The goal of the projection is to increase the accuracy of the conversion by minimizing the number of:
- saturated values (i.e. mapped to the destination type min/max),
- zeroed values (because they are below the smallest number that can be represented by the destination type)

The projection is symmetric (affine), i.e it does not use a zero-point. This makes quantized Tensors
compatible with many operations.

One of the benefit of using a lower-bitwidth representation is that you will be able to take advantage of accelerated operations
for the destination type, which are typically faster than their higher precision equivalents.

The current implementation however falls back to `float32` operations for a lot of operations because of a lack of dedicated kernels
(only `int8` matrix multiplication is available).

Note: integer operations cannot be performed in `float16` as a fallback because this format is very bad at representing
`integer` and will likely lead to overflows in intermediate calculations.

Quanto does not support the conversion of a Tensor using mixed destination types.

### Modules

Quanto provides a generic mechanism to replace torch modules by quanto modules that are able to process quanto tensors.

Quanto modules dynamically convert their weights until a model is frozen, which slows down inference a bit but is
required if the model needs to be tuned.

Biases are not converted because to preserve the accuracy of a typical `addmm` operation, they must be converted with a
scale that is equal to the product of the input and weight scales, which leads to a ridiculously small scale, and conversely
requires a very high bitwidth to avoid clipping. Typically, with `int8` inputs and weights, biases would need to be quantized
with at least `12` bits, i.e in `int16`. Since most biases ar today `float16`, this is a waste of time.

Activations are dynamically quantized using static scales (defaults to the range `[-1, 1]`). The model needs to be calibrated to evaluate the best activation scales (using a momentum).

## Performances

**DISCLAIMER**: These are preliminary observations gathered from a panel of models, and not an actual performance report.

In terms of accuracy:

- models using only int8 weights do not seem to suffer any drop in accuracy,
- models using also int8 activations do suffer from moderate to severe accuracy drops,
- using float8 activations can help in getting a better accuracy.

In terms of speed:

- models using int8 weights only are very slightly slower than the original float model due to the weight dequantization,
- models using int8 activations are slightly slower on CUDA devices,
- models using int8 activations are significantly slower on CPU and MPS devices, where fallbacks are triggered.
- models using float8 activations are significantly slower on CUDA devices, where fallbacks are triggered.

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

**1. Quantize**

The first step converts a standard float model into a dynamically quantized model.

```python
quantize(model, weights=torch.int8, activations=torch.int8)
```

At this stage, only the inference of the model is modified to dynamically quantize the weights.

**2. Calibrate (optional)**

Quanto supports a calibration mode that allows to record the activation ranges while passing representative samples through the quantized model.

```python
with calibration(momentum=0.9):
    model(samples)
```

This automatically activates the quantization of the activations in the quantized modules.


**3. Tune, aka Quantization-Aware-Training (optional)**

If the performance of the model degrades too much, one can tune it for a few epochs to recover the float model performance.

```python
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data).dequantize()
    loss = torch.nn.functional.nll_loss(output, target)
    loss.backward()
    optimizer.step()
```

**4. Freeze integer weights**

When freezing a model, its float weights are replaced by quantized integer weights.

```python
freeze(model)
```

Please refer to the [examples](https://github.com/huggingface/quanto/tree/main/examples) for instantiations of that workflow.

## Per-axis versus per-tensor

Activations are always quantized per-tensor because most linear algebra operations in a model graph are not compatible with per-axis inputs: you simply cannot add numbers that are not expressed in the same base (`you cannot add apples and oranges`).

Weights involved in matrix multiplications are, in the contrary, always quantized along their first axis, because all output features are evaluated independently from one another.

The outputs of a quantized matrix multiplication will anyway always be dequantized, even if activations are quantized, because:

- the resulting integer values are expressed with a much higher bitwidth (typically `int32`) than the activation bitwidth (typically `int8`),
- they might be combined with a `float` bias.

Quantizing activations per-tensor can lead to serious quantization errors if the corresponding tensors contain large outlier values. Typically, this will lead to quantized tensors with most values set to zero (except the outliers).

A possible solution to work around that issue is to 'smooth' the activations statically as illustrated by [SmoothQuant](https://github.com/mit-han-lab/smoothquant). You can find a script to smooth some model architectures under [external/smoothquant](external/smoothquant).

A better option is often to represent activations using `float8` instead of `int8`.
