#!/bin/bash
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPT_PATH=$(dirname "$SCRIPT")

models=(
    google/gemma-2b
    meta-llama/Meta-Llama-3.1-8B
    mistralai/Mistral-7B-Instruct-v0.3
)

for m in ${models[@]}; do
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric prediction --png --json --batch_size 16
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric perplexity --png --json --batch_size 16
    python ${SCRIPT_PATH}/evaluate_configurations.py --model $m --metric latency --png --json --batch_size 16
done
