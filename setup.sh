#!/bin/bash

VENV=${1:-.venv}
if [ ! -d "${VENV}" ]; then
    python3 -m venv .venv
fi
. ${VENV}/bin/activate
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip install black ruff
