#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install miniconda if not present
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n NA-MPNN python=3.10
conda activate NA-MPNN

# Core packages
conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch=2.5.1 pytorch-cuda=12.4 \
    openbabel=3.1.1 \
    numpy=2.4.1 pandas biopython

# Pip packages
pip install ema-pytorch==0.7.9 wandb==0.25.1

echo "Environment ready."
echo "Next: update paths in dfm_model.json and run:"
echo "  torchrun --nproc_per_node=NUM_GPUS ${REPO_ROOT}/na_run.py ${REPO_ROOT}/dfm_model.json"
