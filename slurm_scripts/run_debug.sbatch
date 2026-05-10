#!/bin/bash
#SBATCH -A LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 03:20:00
#SBATCH -J na_mpnn_debug
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /home/mh2167/rds/hpc-work/NA-MPNN
mkdir -p logs

source ~/.bashrc
# Temporarily allow unset vars during conda activation (MKL scripts expect them).
set +u
conda activate NA-MPNN
set -u

python na_run.py debug_model.json
