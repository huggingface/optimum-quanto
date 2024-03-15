# Quanto generation benchmark

This repository contains scripts to evaluate the performances of quantized models using three metrics:

- `latency.py` evaluates the latency per generated token,
- `prediction.py` evaluates the accuracy when predicting the last token of prompts from the [Lambada dataset](https://huggingface.co/datasets/lambada),
- `perplexity.py` evaluates the perplexity of the model on the [WikiText dataset](https://huggingface.co/datasets/wikitext), as defined in the [transformers documentation](https://huggingface.co/docs/transformers/en/perplexity).

A `evaluate_model.py` utility script is also provided to evaluate the metrics on a specific model for several quantization configurations, and output the result to a `png` barchart and/or a `json` file.

The paragraphs below display results for some popular models on a NVIDIA A100 GPU.

## facebook/opt-125m

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/facebook-opt-125m_Accuracy.png" alt="facebook/opt-125m Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/facebook-opt-125m_Perplexity.png" alt="facebook/opt-125m WikiText perplexity">
  </div>
 </center>
</div>

## facebook/opt-350m

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/facebook-opt-350m_Accuracy.png" alt="facebook/opt-350m Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/facebook-opt-350m_Perplexity.png" alt="facebook/opt-350m WikiText perplexity">
  </div>
 </center>
</div>

## facebook/opt-1.3b

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/facebook-opt-1.3b_Accuracy.png" alt="facebook/opt-1.3bm Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/facebook-opt-1.3b_Perplexity.png" alt="facebook/opt-1.3bm WikiText perplexity">
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

## NousResearch/Llama-2-7b-hf

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/NousResearch-Llama-2-7b-hf_Accuracy.png" alt="NousResearch/Llama-2-7b-hf Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/NousResearch-Llama-2-7b-hf_Perplexity.png" alt="NousResearch/Llama-2-7b-hf WikiText perplexity">
  </div>
 </center>
</div>

## mistralai/Mistral-7B-v0.1

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-v0.1_Accuracy.png" alt="mistralai/Mistral-7B-v0.1 Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-v0.1_Perplexity.png" alt="mistralai/Mistral-7B-v0.1 Lambada prediction accuracy">
  </div>
 </center>
</div>

## HuggingFaceH4/zephyr-7b-beta

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/HuggingFaceH4-zephyr-7b-beta_Accuracy.png" alt="HuggingFaceH4/zephyr-7b-beta Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/HuggingFaceH4-zephyr-7b-beta_Perplexity.png" alt="HuggingFaceH4/zephyr-7b-beta Lambada prediction accuracy">
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
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/google-gemma-2b_Perplexity.png" alt="google-gemma-2b Lambada prediction accuracy">
  </div>
 </center>
</div>
