# SmoothQuant original conversion script

This converts an OPT or Bloom [🤗 transformers](https://github.com/huggingface/transformers) model to a "smoothed" version, as described in
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438).

```bash
$ python smoothquant.py --model facebook/opt-1.3b --save-path smoothed-models/facebook/opt-1.3b
```

Note: due to hard-coded assumptions on model architecture in the script this only works for OPT models that apply the layer_norm
before the attention (`do_layer_norm_before=true` in `config.json`). This means all models but `facebook/opt-350m`.
