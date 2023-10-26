# Quantization of a `transformers` causal lm model

The model is quantized and its ability to predict the last token of some samples of the `lambada` dataset is evaluated.

Tested with the following architectures:
- opt -> works, but accuracy is 0 %,
- gpt-neox -> works, but accuracy is 0 %.
