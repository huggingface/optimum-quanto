#!/bin/bash
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPT_PATH=$(dirname "$SCRIPT")

small_models=(
    facebook/opt-125m
    facebook/opt-350m
    facebook/opt-1.3b
    EleutherAI/pythia-1b
    princeton-nlp/Sheared-LLaMA-1.3B
)

bigger_models=(
    NousResearch/Llama-2-7b-hf
    HuggingFaceH4/zephyr-7b-beta
)

for m in ${small_models[@]}; do
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric prediction --png
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric perplexity --png
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric latency --png
done

for m in ${bigger_models[@]}; do
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric prediction --png --json --batch_size 16
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric perplexity --png --json --batch_size 16
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric latency --png --json --batch_size 16
done
