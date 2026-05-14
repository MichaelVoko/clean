#!/bin/bash
#SBATCH -p icelake
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=design_sequences

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
if [ -z "$REPO_ROOT" ] || [ ! -d "$REPO_ROOT/evaluation" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
NA_EVAL_UTILS="${NA_EVAL_UTILS:-${REPO_ROOT}/evaluation/na_eval_utils.py}"

CONDA_ROOT="${CONDA_ROOT:-/home/mh2167/miniconda3}"
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-NA-MPNN}"
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
        echo "Warning: conda environment '$CONDA_ENV_NAME' not found. Using existing Python." >&2
        echo "Set CONDA_ENV_NAME to a valid env name or NA_EVAL_PYTHON_BIN to a specific Python executable." >&2
    fi
fi
PYTHON_BIN="${NA_EVAL_PYTHON_BIN:-$(command -v python)}"

CSV_FILE=$1
OUTPUT_DIR=$2
METHOD=$3
NUM_SAMPLES=$4
TEMPERATURE=${5:-}
NA_MPNN_MODEL_PATH=${6:-}
MODEL_MODE=${7:-}
DFM_DT=${8:-}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

# 1) sanity check
if [[ ! -f "$CSV_FILE" ]]; then
    echo "CSV file '$CSV_FILE' not found!" >&2
    exit 1
fi

# 2) read all structure_path values via Python csv.DictReader
mapfile -t STRUCTURE_PATHS < <(
    "${PYTHON_BIN}" - "$CSV_FILE" <<'PYCODE'
import sys, pandas as pd

df = pd.read_csv(sys.argv[1])

for p in df['structure_path']:
    print(p)
PYCODE
)

total=${#STRUCTURE_PATHS[@]}
if (( total == 0 )); then
    echo "No data rows found in CSV." >&2
    exit 1
fi

# 3) compute chunking based on SLURM array
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total )) && END_IDX=$(( total - 1 ))

# 4) process this shard
for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    structure_path=${STRUCTURE_PATHS[idx]}

    structure_basename=$(basename "$structure_path")
    structure_base_no_gz="${structure_basename%.gz}"
    structure_name="${structure_base_no_gz%.cif}"
    structure_name="${structure_name%.pdb}"
    design_json_dir="$OUTPUT_DIR/$structure_name/design_json"

    if [[ "$SKIP_COMPLETED" == "1" && -d "$design_json_dir" && -n "$(compgen -G "$design_json_dir/*.json")" ]]; then
        echo "Skipping $structure_path (already has design_json outputs in $design_json_dir)"
        continue
    fi

    cmd=(
        "${PYTHON_BIN}" "${NA_EVAL_UTILS}"
        --function_name "design_nucleic_acid_sequence"
        --structure_path "$structure_path"
        --overall_output_directory "$OUTPUT_DIR"
        --num_samples "$NUM_SAMPLES"
        --method "$METHOD"
    )

    if [[ -n "$TEMPERATURE" ]]; then
        cmd+=(--temperature "$TEMPERATURE")
    fi

    if [[ -n "$NA_MPNN_MODEL_PATH" ]]; then
        cmd+=(--na_mpnn_model_path "$NA_MPNN_MODEL_PATH")
    fi

    if [[ -n "$MODEL_MODE" ]]; then
        cmd+=(--model_mode "$MODEL_MODE")
    fi

    if [[ -n "$DFM_DT" ]]; then
        cmd+=(--dfm_dt "$DFM_DT")
    fi

    # Execute the command
    "${cmd[@]}"
done
