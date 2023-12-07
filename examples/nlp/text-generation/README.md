# Quantization of a `transformers` causal lm model

The model is quantized and its ability to predict the last token of some samples of the `lambada` dataset is evaluated.

This is not a very robust test, as we evaluate with the same sentences we use for calibration, but it still provides some useful information.

Note: since the samples are shuffled, the results might be different between runs.

| model                            | fp32 | fp16 | w int8 a fp32 | w int8 a fp16 | w int8 a int8 per-tensor | w int8 a int8 per-axis |
|----------------------------------|------|------|---------------|---------------|--------------------------|------------------------|
| facebook/opt-125m                | 0.61 | 0.61 | 0.61          | 0.61          | 0.05                     | 0.47                   |
| facebook/opt-350m                | 0.63 | 0.63 | 0.63          | 0.63          | **0.59**                 | 0.59                   |
| facebook/opt-1.3b                | 0.72 | 0.72 | 0.72          | 0.72          | 0.53                     | 0.34                   |
| EleutherAI/pythia-160m           | 0.65 | 0.62 | 0.64          | 0.61          | 0.03                     | 0.29                   |
| EleutherAI/pythia-410m           | 0.71 | 0.71 | 0.71          | 0.71          | 0.16                     | 0.21                   |
| EleutherAI/pythia-1b             | 0.75 | 0.75 | 0.75          | 0.75          | 0.50                     | 0.33                   |
| princeton-nlp/Sheared-LLaMA-1.3B | 0.83 | 0.83 | 0.83          | 0.83          | **0.65**                 | 0.73                   |

As we can see, there is no performance degradation when quantizing only the weights to int8.

When quantizing also the activations per-tensor, there are wild variations between models, regardless of their size.

The models that have the lowest per-tensor accuracy can't recover the float accuracy when quantizing per-axis.

Some of the models (**in bold**) are doing extremely well.
