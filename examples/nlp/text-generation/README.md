# Quantization of a `transformers` causal lm model

The model is quantized and its ability to predict the last token of some samples of the `lambada` test dataset is evaluated.

Models using `int8` or `float8` weights are calibrated using `128` samples of the `lambada` validation dataset.

Model activations are streamlined: if the next operation is incompatible with quantized inputs, then the activations are disabled
to avoid a spurious quantize/dequantize that only degrades accuracy.

Note: since the calibration samples are shuffled, the results might be slightly different between runs if you change the seed.

| model                            | fp16 | w int8 a fp16 | w int8 a int8 | w int8 a fp8_e5m2 | w int8 a fp8_e4m3 |
|----------------------------------|------|---------------|---------------|-------------------|-------------------|
| facebook/opt-125m                | 0.63 | 0.63          | 0.06          | 0.00              | **0.61**          |
| facebook/opt-350m                | 0.67 | 0.67          | **0.64**      | 0.00              | **0.65**          |
| facebook/opt-1.3b                | 0.76 | 0.76          | 0.18          | 0.00              | **0.70**          |
| EleutherAI/pythia-160m           | 0.44 | 0.44          | 0.00          | 0.00              | 0.00              |
| EleutherAI/pythia-410m           | 0.68 | 0.68          | 0.43          | 0.22              | 0.41              |
| EleutherAI/pythia-1b             | 0.71 | 0.72          | **0.70**      | **0.69**          | **0.71**          |
| princeton-nlp/Sheared-LLaMA-1.3B | 0.83 | 0.83          | **0.81**      | **0.72**          | **0.80**          |
| 01-ai/Yi-6B (bfloat 16)          | 0.83 | 0.83          | **0.80**      | 0.00              | **0.76**          |
| NousResearch/Llama-2-7b-hf       | 0.89 | 0.89          | **0.87**      | **0.76**          | **0.85**          |
| HuggingFaceH4/zephyr-7b-beta     | 0.86 | 0.86          | **0.85**      | 0.00              | **0.83**          |

As we can see, there is no performance degradation when quantizing only the weights to int8.

When quantizing also the activations, there are wild variations between models, but we can see a pattern:
- smaller models are generally more sensitive to 8-bit activations,
- for models whose accuracy is degraded, using float8 activations often gives better result.

Some of the models (**in bold**) are doing extremely well.
