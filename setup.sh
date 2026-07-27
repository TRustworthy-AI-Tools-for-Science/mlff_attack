#!/usr/bin/env bash

set -e

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: Run this inside Ubuntu, Linux, or WSL."
    exit 1
fi

sudo apt update
sudo apt install -y python3.12 python3.12-venv

if [ ! -x .venv-uma/bin/python ]; then python3.12 -m venv .venv-uma; fi

.venv-uma/bin/python -m pip install --upgrade pip
.venv-uma/bin/python -m pip install -e ".[uma]"

source .venv-uma/bin/activate
echo "UMA setup complete: `source .venv-mace/bin/activate` to activate."
echo "Authenticate if required: .venv-uma/bin/hf auth login"
