#!/usr/bin/env bash

set -e

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: Run this inside Ubuntu, Linux, or WSL."
    exit 1
fi

sudo apt update
sudo apt install -y python3.12 python3.12-venv

if [ ! -x .venv-chgnet/bin/python ]; then python3.12 -m venv .venv-chgnet; fi

.venv-chgnet/bin/python -m pip install --upgrade pip
.venv-chgnet/bin/python -m pip install -e ".[chgnet]"

source .venv-chgnet/bin/activate
echo "CHGNet setup complete. Activate it with: source .venv-chgnet/bin/activate"
