# Quantization of a `transformers` causal lm model

The model is quantized and its ability to predict the last token of some samples of the `lambada` dataset is evaluated.

Note: since the samples are shuffled, the results might be different between runs.

| model                            | fp16 | w int8 a fp16 | w int8 a int8 | w int8 a fp8_e5m2 | w int8 a fp8_e4m3 |
|----------------------------------|------|---------------|---------------|-------------------|-------------------|
| facebook/opt-125m                | 0.58 | 0.58          | 0.04          | 0.00              | 0.56              |
| facebook/opt-350m                | 0.62 | 0.63          | **0.60**      | 0.00              | **0.60**          |
| facebook/opt-1.3b                | 0.71 | 0.71          | 0.48          | 0.00              | **0.66**          |
| EleutherAI/pythia-160m           | 0.47 | 0.46          | 0.00          | 0.00              | 0.00              |
| EleutherAI/pythia-410m           | 0.65 | 0.64          | 0.11          | 0.03              | 0.32              |
| EleutherAI/pythia-1b             | 0.70 | 0.70          | 0.43          | 0.42              | 0.66              |
| princeton-nlp/Sheared-LLaMA-1.3B | 0.83 | 0.83          | 0.64          | 0.72              | **0.77**          |
| 01-ai/Yi-6B (bfloat 16)          | 0.81 | 0.81          | 0.11          | 0.66              | 0.47              |
| NousResearch/Llama-2-7b-hf       | 0.88 | 0.88          | **0.76**      | 0.04              | **0.81**          |
| HuggingFaceH4/zephyr-7b-beta     | 0.85 | 0.85          | 0.31          | **0.77**          | **0.76**          |

As we can see, there is no performance degradation when quantizing only the weights to int8.

When quantizing also the activations, there are wild variations between models, but we can see a pattern:
- smaller models are generally more sensitive to activations quantization,
- for models whose accuracy is degraded, using float8 activations always gives better result.

Some of the models (**in bold**) are doing extremely well.
