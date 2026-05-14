#!/bin/bash
#SBATCH -p icelake
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=score_designs

# Activate conda environment
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/.bashrc" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.bashrc"
    fi
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found. Ensure conda is available in PATH before running this script."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate NA-MPNN

SCRIPT_DIR="/rds/user/mh2167/hpc-work/NA-MPNN/evaluation/rna_design_scripts"
REPO_ROOT="/rds/user/mh2167/hpc-work/NA-MPNN"
NA_EVAL_UTILS="${NA_EVAL_UTILS:-${REPO_ROOT}/evaluation/na_eval_utils.py}"
PYTHON_BIN="${NA_EVAL_PYTHON_BIN:-python}"

PROCESSED_DIR=$1
PROCESSED_REF_DIR=$2
OUTPUT_DIR=$3

# --- 1) Sanity checks ---
if [[ ! -d "$PROCESSED_DIR" ]]; then
    echo "Processed designs directory '$PROCESSED_DIR' not found!" >&2
    exit 1
fi
if [[ ! -d "$PROCESSED_REF_DIR" ]]; then
    echo "Processed reference directory '$PROCESSED_REF_DIR' not found!" >&2
    exit 1
fi
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Score output directory '$OUTPUT_DIR' not found; creating it..."
    mkdir -p "$OUTPUT_DIR"
fi

# 2) Collect all JSON files under the processed-designs directory
shopt -s nullglob
json_files=( "$PROCESSED_DIR"/*/processed_design_json/*.json )
total_json=${#json_files[@]}
if (( total_json == 0 )); then
    echo "No JSON files found under $PROCESSED_DIR/*/processed_design_json/*.json." >&2
    exit 1
fi

# 3) Compute chunk boundaries for this SLURM array task
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total_json + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total_json )) && END_IDX=$(( total_json - 1 ))

# 4) Process the assigned slice of JSON files
for idx in $(seq "$START_IDX" "$END_IDX"); do
    json_path=${json_files[idx]}
    filename=$(basename "$json_path" .json)
    pdb_id="${filename%_*}"

    ref_json="$PROCESSED_REF_DIR/$pdb_id/reference_json/$pdb_id.json"

    if [[ ! -f "$ref_json" ]]; then
        echo "Reference JSON not found for $pdb_id: '$ref_json'. Skipping." >&2
        continue
    fi

    # Skip if this design has already been scored.
    score_output="$OUTPUT_DIR/$filename/$filename.json"
    if [[ -f "$score_output" ]]; then
        echo "Already scored: $filename. Skipping."
        continue
    fi

    "${PYTHON_BIN}" "${NA_EVAL_UTILS}" \
            --function_name score_design_monomer_rna \
            --reference_path "$ref_json" \
            --subject_path "$json_path" \
            --overall_output_directory "$OUTPUT_DIR"
done
