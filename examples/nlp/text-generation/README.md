# Quantization of a `transformers` causal lm model

The model is quantized and its ability to predict the last token of some samples of the `lambada` dataset is evaluated.

This is not a very robust test, as we evaluate with the same sentences we use for calibration, but it still provides some useful information.

Note: since the samples are shuffled, the results might be different between runs.

|  model                           |  float |  per-tensor |  per-axis |
|----------------------------------|--------|-------------|-----------|
| facebook/opt-125m                | 0.61   | 0.03        | 0.59      |
| facebook/opt-350m                | 0.63   | **0.63**    | 0.63      |
| facebook/opt-1.3b                | 0.72   | 0.46        | 0.72      |
| EleutherAI/pythia-160m           | 0.65   | 0.04        | 0.62      |
| EleutherAI/pythia-410m           | 0.71   | 0.19        | 0.56      |
| EleutherAI/pythia-1b             | 0.75   | 0.42        | 0.65      |
| princeton-nlp/Sheared-LLaMA-1.3B | 0.83   | **0.66**    | 0.75      |
| NousResearch/Llama-2-7b-hf       | 0.92   | 0.54        | 0.66      |

As we can see, when quantizing per-tensor, there are wild variations between models, regardless of their size.

The models that have the lowest per-tensor accuracy can't recover the float accuracy when quantizing per-axis.

Some of the models (**in bold**) are doing extremely well.
