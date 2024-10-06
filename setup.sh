#!/bin/bash

NIGHTLY=${1:-0}
VENV=".venv"
if [ ! -d "${VENV}" ]; then
    python3 -m venv ${VENV}
fi
. ${VENV}/bin/activate
if [ "$NIGHTLY" -eq "0" ]; then
    pip install --upgrade torch torchvision torchaudio
else
    pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
fi
# Build tools
pip install ruff pytest build
# For examples
pip install accelerate transformers datasets
