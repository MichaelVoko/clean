#!/usr/bin/env bash
set -euo pipefail

# Initialize conda for non-interactive shells (srun/sbatch).
source /home/mh2167/miniconda3/etc/profile.d/conda.sh
# Temporarily relax nounset for conda activation scripts.
set +u
conda activate NA-MPNN
set -u

# Run NA-MPNN preprocessing on a compute node via srun.
# Skips structures that already have all expected outputs.
# Prefer explicit override, then SLURM submit directory, then script-relative path.
REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "${REPO_ROOT}" && -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}/pdbx:${REPO_ROOT}:${PYTHONPATH:-}"

# Keep temporary files on RDS to avoid node-local /tmp exhaustion.
TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
mkdir -p "${TMPDIR}"
export TMPDIR

# Inputs (override via env vars if needed)
INPUT_CSV="${INPUT_CSV:-${REPO_ROOT}/data/datasets/design_from_splits/preprocessing_input.csv}"
PREPROC_DIR="${PREPROC_DIR:-${REPO_ROOT}/data/datasets/design_from_splits/preprocessed_data}"
FILTERED_CSV="${FILTERED_CSV:-${REPO_ROOT}/data/datasets/design_from_splits/preprocessing_input.todo.csv}"
N_PREPROC="${N_PREPROC:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "TMPDIR=${TMPDIR}"
echo "INPUT_CSV=${INPUT_CSV}"
echo "PREPROC_DIR=${PREPROC_DIR}"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV not found: ${INPUT_CSV}" >&2
  exit 1
fi

mkdir -p "${PREPROC_DIR}"

echo "Filtering already-processed entries..."
INPUT_CSV="${INPUT_CSV}" PREPROC_DIR="${PREPROC_DIR}" FILTERED_CSV="${FILTERED_CSV}" \
python - <<'PY'
import csv
import os
import sys

input_csv = os.environ["INPUT_CSV"]
preproc_dir = os.environ["PREPROC_DIR"]
filtered_csv = os.environ["FILTERED_CSV"]

required = [
    ("sequences", ".csv"),
    ("asmb_lengths", ".npy"),
    ("asmb_interface_masks", ".npy"),
    ("asmb_side_chain_interface_masks", ".npy"),
    ("asmb_nearest_protein_side_chain_index", ".npy"),
    ("asmb_base_pair_masks", ".npy"),
    ("asmb_base_pair_index", ".npy"),
    ("asmb_canonical_base_pair_masks", ".npy"),
    ("asmb_canonical_base_pair_index", ".npy"),
]

def structure_name(path):
    base = os.path.basename(path)
    if base.endswith(".gz"):
        base = os.path.splitext(os.path.splitext(base)[0])[0]
    else:
        base = os.path.splitext(base)[0]
    return base

with open(input_csv, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames or []

if not fieldnames:
    print("Input CSV has no header/columns.", file=sys.stderr)
    sys.exit(1)

todo = []
for row in rows:
    name = structure_name(row["structure_path"])
    missing = False
    for subdir, ext in required:
        path = os.path.join(preproc_dir, subdir, name + ext)
        if not os.path.exists(path):
            missing = True
            break
    if missing:
        todo.append(row)

with open(filtered_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(todo)

print(f"{len(todo)} remaining / {len(rows)} total -> {filtered_csv}")
PY

remaining=$(awk 'NR>1{c++} END{print c+0}' "${FILTERED_CSV}")
if [[ "${remaining}" -eq 0 ]]; then
  echo "All entries already processed. Nothing to do."
  exit 0
fi

if (( N_PREPROC > remaining )); then
  N_PREPROC="${remaining}"
fi
if (( N_PREPROC < 1 )); then
  N_PREPROC=1
fi

echo "Starting preprocessing with ${N_PREPROC} workers..."
pids=()
for r in $(seq 0 $((N_PREPROC - 1))); do
  python "${REPO_ROOT}/data/preprocess_dataset.py" \
    "${FILTERED_CSV}" \
    "${PREPROC_DIR}" \
    "${N_PREPROC}" \
    "${r}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}"; done

echo "Done."
