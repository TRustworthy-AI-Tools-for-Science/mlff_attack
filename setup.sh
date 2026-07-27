#!/usr/bin/env bash

set -e

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: Run this inside Ubuntu, Linux, or WSL."
    exit 1
fi

if ! command -v apt >/dev/null 2>&1; then
    echo "ERROR: This setup requires Ubuntu or WSL Ubuntu."
    exit 1
fi

sudo apt update
sudo apt install -y python3 python3-venv

if [ ! -x .venv-mace/bin/python ]; then
    python3 -m venv .venv-mace
fi

.venv-mace/bin/python -m pip install --upgrade pip
.venv-mace/bin/python -m pip install -e ".[mace]"

source .venv-mace/bin/activate
echo "MACE setup complete. Activate it with: source .venv-mace/bin/activate"
