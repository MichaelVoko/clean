#!/bin/bash
#SBATCH -A LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32g
#SBATCH -t 00:30:00
#SBATCH -J entropy_est
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eo pipefail

REPO_ROOT="/home/mh2167/rds/hpc-work/NA-MPNN"
cd "$REPO_ROOT"

CONDA_ROOT="${CONDA_ROOT:-/home/mh2167/miniconda3}"
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate NA-MPNN
set -u

CKPT="${CKPT:-dfm_base/s_54019.pt}"
N_STRUCT="${N_STRUCT:-64}"
N_TIME="${N_TIME:-1024}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/evaluation/entropy}"

python evaluation/entropy/estimate_na_entropy.py \
    --checkpoint "$CKPT" \
    --csv-dir evaluation/sweeps/scripts/valid_datasets \
    --n-structures "$N_STRUCT" \
    --n-time "$N_TIME" \
    --out-dir "$OUT_DIR" \
    --device cuda
