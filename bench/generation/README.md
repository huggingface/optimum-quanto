# Quanto generation benchmark

This repository contains scripts to evaluate the performances of quantized models using three metrics:

- `latency.py` evaluates the latency per generated token,
- `prediction.py` evaluates the accuracy when predicting the last token of prompts from the [Lambada dataset](https://huggingface.co/datasets/lambada),
- `perplexity.py` evaluates the perplexity of the model on the [WikiText dataset](https://huggingface.co/datasets/wikitext), as defined in the [transformers documentation](https://huggingface.co/docs/transformers/en/perplexity).

A `evaluate_model.py` utility script is also provided to evaluate the metrics on a specific model for several quantization configurations, and output the result to a `png` barchart and/or a `json` file.

Note: the language modeling head (lm_head) of the tested models is not quantized.

The paragraphs below display results for some popular models on a NVIDIA A10 GPU.

## meta-llama/Meta-Llama-3-8B

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3-8B_Accuracy.png" alt="meta-llama/Meta-llama-3-8B Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3-8B_Perplexity.png" alt="meta-llama/Meta-Llama-3-8B WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/meta-llama-Meta-Llama-3-8B_Latency__ms_.png" alt="meta-llama/Meta-Llama-3-8B Latency">
  </div>
 </center>
</div>

## mistralai/Mistral-7B-Instruct-v0.3

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-Instruct-v0.3_Accuracy.png" alt="mistralai/Mistral-7B-Instruct-v0.3 Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-Instruct-v0.3_Perplexity.png" alt="mistralai/Mistral-7B-Instruct-v0.3 WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-Instruct-v0.3_Latency__ms_.png" alt="mistralai/Mistral-7B-Instruct-v0.3 Latency">
  </div>
 </center>
</div>

## google/gemma-2b

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/google-gemma-2b_Accuracy.png" alt="google-gemma-2b Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/google-gemma-2b_Perplexity.png" alt="google-gemma-2b WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/google-gemma-2b_Latency__ms_.png" alt="google-gemma-2b Latency">
  </div>
 </center>
</div>

## EleutherAI-pythia-1b

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/EleutherAI-pythia-1b_Accuracy.png" alt="EleutherAI-pythia-1b Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/EleutherAI-pythia-1b_Perplexity.png" alt="EleutherAI-pythia-1b WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/EleutherAI-pythia-1b_Latency__ms_.png" alt="EleutherAI-pythia-1b Latency">
  </div>
 </center>
</div>

## princeton-nlp/Sheared-LLaMA-1.3B

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/princeton-nlp-Sheared-LLaMA-1.3B_Accuracy.png" alt="princeton-nlp/Sheared-LLaMA-1.3B Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/princeton-nlp-Sheared-LLaMA-1.3B_Perplexity.png" alt="princeton-nlp/Sheared-LLaMA-1.3B WikiText perplexity">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/princeton-nlp-Sheared-LLaMA-1.3B_Latency__ms_.png" alt="princeton-nlp/Sheared-LLaMA-1.3B Latency">
  </div>
 </center>
</div>
