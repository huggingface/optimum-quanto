# Optimum Quanto

🤗 Optimum Quanto is a python quantization backend for [optimum](https://huggingface.co/docs/optimum/en/index) that provides several features that are either not supported or limited by the base [pytorch quantization tools](https://pytorch.org/docs/stable/quantization.html):

- all features are available in eager mode (works with non-traceable models),
- quantized models can be placed on any device (including CUDA and MPS),
- automatically inserts quantization and dequantization stubs,
- automatically inserts quantized functional operations,
- automatically inserts quantized modules (see below the list of supported modules),
- provides a seamless workflow from a float model to a dynamic to a static quantized model,
- serialization compatible with pytorch `weight_only` and 🤗 `safetensors`,
- accelerated matrix multiplications on CUDA devices (int8-int8, fp16-int4),
- supports int2, int4, int8 and float8 weights,
- supports int8 and float8 activations.

Features yet to be implemented:

- dynamic activations smoothing,
- kernels for all mixed matrix multiplications on all devices,
- compatibility with [torch compiler](https://pytorch.org/docs/stable/torch.compiler.html) (aka dynamo).

## Quantized modules

Thanks to a seamless propagation mechanism through quantized tensors, only a few modules working as quantized
tensors insertion points are actually required.

The following modules can be quantized:

- [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (QLinear).
Weights are always quantized, and biases are not quantized. Inputs and outputs can be quantized.
- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) (QConv2D).
Weights are always quantized, and biases are not quantized. Inputs and outputs can be quantized.
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
Weights and biases are __not__ quantized. Outputs can be quantized.

## Limitations and design choices

### Tensors

At the heart of quanto is a Tensor subclass that corresponds to:
- the projection of a source Tensor into the optimal range for a given destination type,
- the mapping of projected values to the destination type.

For floating-point destination types, the mapping is done by the native pytorch cast (i.e. `Tensor.to()`).

For integer destination types, the mapping is a simple rounding operation (i.e. `torch.round()`).

The goal of the projection is to increase the accuracy of the conversion by minimizing the number of:
- saturated values (i.e. mapped to the destination type min/max),
- zeroed values (because they are below the smallest number that can be represented by the destination type)

The projection is symmetric (affine), i.e. it does not use a zero-point. This makes quantized Tensors
compatible with many operations.

One of the benefits of using a lower-bitwidth representation is that you will be able to take advantage of accelerated operations
for the destination type, which is typically faster than their higher precision equivalents.

The current implementation however falls back to `float32` operations for a lot of operations because of a lack of dedicated kernels
(only `int8` matrix multiplication is available).

Note: integer operations cannot be performed in `float16` as a fallback because this format is very bad at representing
`integer` and will likely lead to overflows in intermediate calculations.

Quanto does not support the conversion of a Tensor using mixed destination types.

### Modules

Quanto provides a generic mechanism to replace `torch` modules by `optimum-quanto` modules that are able to process quanto tensors.


`optimum-quanto` modules dynamically convert their weights until a model is frozen, which slows down inference a bit but is
required if the model needs to be tuned.

Biases are not converted because to preserve the accuracy of a typical `addmm` operation, they must be converted with a
scale that is equal to the product of the input and weight scales, which leads to a ridiculously small scale, and conversely
requires a very high bitwidth to avoid clipping. Typically, with `int8` inputs and weights, biases would need to be quantized
with at least `12` bits, i.e. in `int16`. Since most biases are today `float16`, this is a waste of time.

Activations are dynamically quantized using static scales (defaults to the range `[-1, 1]`). The model needs to be calibrated to evaluate the best activation scales (using a momentum).

## Performances

In a nutshell:

- accuracy: models compiled with `int8`/`float8` weights and `float8` activations are very close to the `16-bit` models,
- latency: all models but those using int4 weights are at least `2x` slower than the `16-bit` models due to the lack of optimized kernels (for now).
- device memory: approximately divided by float bits / integer bits.

The paragraph below is just an example. Please refer to the `bench` folder for detailed results per use-case of model.

### meta-llama/Meta-Llama-3-8B

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/optimum-quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3-8B_Perplexity.png" alt="meta-llama/Meta-Llama-3-8B WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/optimum-quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3-8B_Latency__ms_.png" alt="meta-llama/Meta-Llama-3-8B Latency">
  </div>
 </center>
</div>

## Installation

Optimum Quanto is available as a pip package.

```sh
pip install optimum-quanto
```

## Quantization workflow for vanilla pytorch models

Quanto does not make a clear distinction between dynamic and static quantization: models are always dynamically quantized,
but their weights can later be "frozen" to integer values.

A typical quantization workflow would consist of the following steps:

**1. Quantize**

The first step converts a standard float model into a dynamically quantized model.

```python
from optimum.quanto import quantize, qint8

quantize(model, weights=qint8, activations=qint8)
```

At this stage, only the inference of the model is modified to dynamically quantize the weights.

**2. Calibrate (optional if activations are not quantized)**

Quanto supports a calibration mode that allows to record the activation ranges while passing representative samples through the quantized model.

```python
from optimum.quanto import Calibration

with Calibration(momentum=0.9):
    model(samples)
```

This automatically activates the quantization of the activations in the quantized modules.


**3. Tune, aka Quantization-Aware-Training (optional)**

If the performance of the model degrades too much, one can tune it for a few epochs to recover the float model performance.

```python
import torch

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
from optimum.quanto import freeze

freeze(model)
```

**5. Serialize quantized model**

Quantized models weights can be serialized to a `state_dict`, and saved to a file.
Both `pickle` and `safetensors` (recommended) are supported.

```python
from safetensors.torch import save_file

save_file(model.state_dict(), 'model.safetensors')
```

In order to be able to reload these weights, you also need to store the quantized
model quantization map.

```python
import json

from optimum.quanto import quantization_map

with open('quantization_map.json', w) as f:
  json.dump(quantization_map(model))
```

**5. Reload a quantized model**

A serialized quantized model can be reloaded from a `state_dict` and a `quantization_map` using the `requantize` helper. Note that you need first to instantiate an empty model.

```python
import json

from safetensors.torch import load_file

state_dict = load_file('model.safetensors')
with open('quantization_map.json', r) as f:
  quantization_map = json.load(f)

# Create an empty model from your modeling code and requantize it
new_model = ...
requantize(new_model, state_dict, quantization_map)
```

Please refer to the [examples](https://github.com/huggingface/quanto/tree/main/examples) for instantiations of that workflow.

## Quantization workflow for Hugging Face models

`optimum-quanto` provides helper classes to serialize/deserialize Hugging Face quantized models.

### LLM models

The first step is to quantize the model

```python
from transformers import AutoModelForCausalLM
from optimum.quanto import QuantizedModelForCausalLM, qint4

model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')
qmodel = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude='lm_head')
```

Note: the model quantized weights will be frozen. If you want to keep them unfrozen to train them you need to use `optimum.quanto.quantize` directly.

The quantized model can be saved using `save_pretrained`:

```python
qmodel.save_pretrained('./Llama-3-8B-quantized')
```

It can later be reloaded using `from_pretrained`:

```python
from optimum.quanto import QuantizedModelForCausalLM

qmodel = QuantizedModelForCausalLM.from_pretrained('Llama-3-8B-quantized')
```

## Per-axis versus per-tensor

Activations are always quantized per-tensor because most linear algebra operations in a model graph are not compatible with per-axis inputs: you simply cannot add numbers that are not expressed in the same base (`you cannot add apples and oranges`).

Weights involved in matrix multiplications are, on the contrary, always quantized along their first axis, because all output features are evaluated independently from one another.

The outputs of a quantized matrix multiplication will anyway always be dequantized, even if activations are quantized, because:

- the resulting integer values are expressed with a much higher bitwidth (typically `int32`) than the activation bitwidth (typically `int8`),
- they might be combined with a `float` bias.

Quantizing activations per-tensor to `int8` can lead to serious quantization errors if the corresponding tensors contain large outlier values. Typically, this will lead to quantized tensors with most values set to zero (except the outliers).

A possible solution to work around that issue is to 'smooth' the activations statically as illustrated by [SmoothQuant](https://github.com/mit-han-lab/smoothquant). You can find a script to smooth some model architectures under [external/smoothquant](external/smoothquant).

A better option is to represent activations using `float8`.
