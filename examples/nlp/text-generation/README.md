# Quantization of a `transformers` causal lm model

The model is quantized and its ability to predict the last token of some samples of the `lambada` test dataset is evaluated.

Models using `int8` or `float8` weights are calibrated using `128` samples of the `lambada` validation dataset.

Note: since the calibration samples are shuffled, the results might be slightly different between runs.

| model                            | fp16 | w int8 a fp16 | w int8 a int8 | w int8 a fp8_e5m2 | w int8 a fp8_e4m3 |
|----------------------------------|------|---------------|---------------|-------------------|-------------------|
| facebook/opt-125m                | 0.63 | 0.63          | 0.05          | 0.01              | **0.61**          |
| facebook/opt-350m                | 0.67 | 0.67          | **0.63**      | 0.00              | **0.64**          |
| facebook/opt-1.3b                | 0.76 | 0.76          | 0.51          | 0.00              | **0.71**          |
| EleutherAI/pythia-160m           | 0.44 | 0.44          | 0.00          | 0.00              | 0.00              |
| EleutherAI/pythia-410m           | 0.68 | 0.68          | 0.12          | 0.05              | 0.32              |
| EleutherAI/pythia-1b             | 0.71 | 0.72          | 0.42          | 0.45              | **0.67**          |
| princeton-nlp/Sheared-LLaMA-1.3B | 0.83 | 0.83          | **0.71**      | **0.75**          | **0.76**          |
| 01-ai/Yi-6B (bfloat 16)          | 0.82 | 0.82          | 0.25          | 0.68              | 0.56              |
| NousResearch/Llama-2-7b-hf       | 0.89 | 0.89          | 0.77          | 0.04              | **0.81**          |
| HuggingFaceH4/zephyr-7b-beta     | 0.86 | 0.86          | 0.31          | **0.78**          | **0.77**          |

As we can see, there is no performance degradation when quantizing only the weights to int8.

When quantizing also the activations, there are wild variations between models, but we can see a pattern:
- smaller models are generally more sensitive to 8-bit activations,
- for models whose accuracy is degraded, using float8 activations always gives better result.

Some of the models (**in bold**) are doing extremely well.
