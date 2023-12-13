#!/bin/bash
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPT_PATH=$(dirname "$SCRIPT")

models=(
    facebook/opt-125m
    facebook/opt-350m
    facebook/opt-1.3b
    EleutherAI/pythia-160m
    EleutherAI/pythia-410m
    EleutherAI/pythia-1b
    princeton-nlp/Sheared-LLaMA-1.3B
)

for m in ${models[@]}; do
    echo $m "w: int8 a: none"
    python ${SCRIPT_PATH}/quantize_causal_lm_model.py --model $m --weights int8 --activations none
    echo $m "w: int8 a: int8"
    python ${SCRIPT_PATH}/quantize_causal_lm_model.py --model $m --weights int8 --activations int8
done
