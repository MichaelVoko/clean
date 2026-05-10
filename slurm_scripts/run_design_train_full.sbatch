#!/usr/bin/env bash
#SBATCH -A LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 2:00:00
#SBATCH -J na_design_train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

 salloc -A LIO-SL3-GPU -p ampere -t 2:00:00 -N 1 --gres=gpu:1 -c 12 --mem=64g
set -euo pipefail

# Resolve repo root: explicit override, then submit dir git root, then script-relative.
REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "${REPO_ROOT}" && -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs

# Keep temporary files off node-local /tmp.
TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
mkdir -p "${TMPDIR}"
export TMPDIR

# Avoid thread oversubscription inside data loading / math kernels.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

MODEL_JSON="${1:-${MODEL_JSON:-${REPO_ROOT}/design_model.json}}"
if [[ ! -f "${MODEL_JSON}" ]]; then
  echo "Training config not found: ${MODEL_JSON}" >&2
  exit 1
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL_JSON=${MODEL_JSON}"
echo "TMPDIR=${TMPDIR}"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

source /home/mh2167/miniconda3/etc/profile.d/conda.sh
set +u
conda activate NA-MPNN
set -u
python --version
nvidia-smi || true

# Some submission contexts leak a stale SLURM_CPUS_PER_TASK (e.g. 32) into this
# job environment. If it disagrees with this job's cpu TRES (e.g. cpu:12), srun
# aborts before launching.
if [[ -n "${SLURM_TRES_PER_TASK:-}" && -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  cpu_from_tres="${SLURM_TRES_PER_TASK#cpu:}"
  cpu_from_tres="${cpu_from_tres%%,*}"
  if [[ -n "${cpu_from_tres}" && "${SLURM_CPUS_PER_TASK}" != "${cpu_from_tres}" ]]; then
    echo "Adjusting SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK} -> ${cpu_from_tres} to match SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK}"
    export SLURM_CPUS_PER_TASK="${cpu_from_tres}"
  fi
fi

srun python "${REPO_ROOT}/na_run.py" "${MODEL_JSON}"
