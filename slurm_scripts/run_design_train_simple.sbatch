#!/usr/bin/env bash
#SBATCH -A LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=64g
#SBATCH -t 12:00:00
#SBATCH -J na_design_train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/rds/user/mh2167/hpc-work/NA-MPNN"
MODEL_JSON="${1:-${REPO_ROOT}/design_model.json}"

if [[ ! -f "${MODEL_JSON}" ]]; then
  echo "Model JSON not found: ${MODEL_JSON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p logs .tmp

export TMPDIR="${REPO_ROOT}/.tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source /home/mh2167/miniconda3/etc/profile.d/conda.sh
conda activate NA-MPNN

echo "Host: $(hostname)"
echo "Model JSON: ${MODEL_JSON}"
python3 "${REPO_ROOT}/na_run.py" "${MODEL_JSON}"
