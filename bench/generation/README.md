# Quanto generation benchmark

This repository contains scripts to evaluate the performances of quantized models using three metrics:

- `latency.py` evaluates the latency per generated token,
- `prediction.py` evaluates the accuracy when predicting the last token of prompts from the [Lambada dataset](https://huggingface.co/datasets/lambada),
- `perplexity.py` evaluates the perplexity of the model on the [WikiText dataset](https://huggingface.co/datasets/wikitext), as defined in the [transformers documentation](https://huggingface.co/docs/transformers/en/perplexity).

A `evaluate_model.py` utility script is also provided to evaluate the metrics on a specific model for several quantization configurations, and output the result to a `png` barchart and/or a `json` file.

Note: the language modeling head (lm_head) of the tested models is not quantized.

The paragraphs below display results for some popular models on a NVIDIA A10 GPU.

## meta-llama/Meta-Llama-3.1-8B

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3.1-8B_bf16_Accuracy.png" alt="meta-llama/Meta-llama-3.1-8B Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3.1-8B_bf16_Perplexity.png" alt="meta-llama/Meta-Llama-3.1-8B WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3.1-8B_bf16_Latency__ms_.png" alt="meta-llama/Meta-Llama-3.1-8B Latency">
  </div>
 </center>
</div>

## mistralai/Mistral-7B-Instruct-v0.3

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-Instruct-v0.3_bf16_Accuracy.png" alt="mistralai/Mistral-7B-Instruct-v0.3 Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-Instruct-v0.3_bf16_Perplexity.png" alt="mistralai/Mistral-7B-Instruct-v0.3 WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-Instruct-v0.3_bf16_Latency__ms_.png" alt="mistralai/Mistral-7B-Instruct-v0.3 Latency">
  </div>
 </center>
</div>

## google/gemma-2b

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/google-gemma-2b_bf16_Accuracy.png" alt="google-gemma-2b Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/google-gemma-2b_bf16_Perplexity.png" alt="google-gemma-2b WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/google-gemma-2b_bf16_Latency__ms_.png" alt="google-gemma-2b Latency">
  </div>
 </center>
</div>
