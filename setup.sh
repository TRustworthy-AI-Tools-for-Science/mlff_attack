#!/usr/bin/env bash

set -e

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: Run this inside Ubuntu, Linux, or WSL."
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Install Miniforge and run setup again."
    echo "https://github.com/conda-forge/miniforge"
    exit 1
fi

sudo apt update
sudo apt install -y git make gcc g++ gfortran

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE '^[[:space:]]*env-mtp[[:space:]]'; then
    conda install --name env-mtp python=3.12 pip -c conda-forge --yes
else
    conda create --name env-mtp python=3.12 pip -c conda-forge --yes
fi

conda activate env-mtp
python -m pip install --upgrade pip
python -m pip install -e .

if [ ! -d mlip-3/.git ]; then git clone https://gitlab.com/ashapeev/mlip-3.git mlip-3; fi

if [ ! -x mlip-3/bin/mlp ]; then
    cd mlip-3
    ./configure --no-mpi --compiler=gnu --blas=embedded
    make mlp
    cd ..
fi

ln -sf "$(pwd)/mlip-3/bin/mlp" "$CONDA_PREFIX/bin/mlp"

conda activate env-mtp
echo "MTP setup complete. Activate it with: conda activate env-mtp"
