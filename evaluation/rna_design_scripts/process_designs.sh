#!/bin/bash
#SBATCH -A LIO-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
#SBATCH --output=/rds/user/mh2167/hpc-work/NA-MPNN/logs/%A_%a.out
#SBATCH --error=/rds/user/mh2167/hpc-work/NA-MPNN/logs/%A_%a.err
#SBATCH --job-name=process_designs

source /home/mh2167/miniconda3/etc/profile.d/conda.sh
conda activate NA-MPNN

# Prepend system lib dir so contrafold subprocess uses the generic x86-64
# libstdc++ rather than the spack Cascadelake-optimised one, which contains
# AVX-512 instructions that cause SIGILL on the AMD EPYC (Milan) cores of the
# ampere nodes.
export LD_LIBRARY_PATH=/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

REPO_ROOT="/rds/user/mh2167/hpc-work/NA-MPNN"
cd "$REPO_ROOT"
NA_EVAL_UTILS="${NA_EVAL_UTILS:-${REPO_ROOT}/evaluation/na_eval_utils.py}"
PYTHON_BIN="${NA_EVAL_PYTHON_BIN:-python}"
SPECIFIED_DIRECTORY=$1
OUTPUT_DIR=$2
SKIP_ALPHAFOLD3=${3:-""}
# 1) sanity checks
if [[ ! -d "$SPECIFIED_DIRECTORY" ]]; then
    echo "Directory '$SPECIFIED_DIRECTORY' not found!" >&2
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "Error: jq required but not on PATH." >&2
    exit 1
fi

# Get the list of JSON files.
shopt -s nullglob
json_files=( "$SPECIFIED_DIRECTORY"/*/design_json/*.json )
total_json=${#json_files[@]}
if (( total_json == 0 )); then
    echo "No JSON files found under $SPECIFIED_DIRECTORY/*/design_json/." >&2
    exit 1
fi

# Number of data rows (excluding the header)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total_json + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total_json )) && END_IDX=$(( total_json - 1 ))

# Collect the assigned chunk of JSON files.
chunk_files=()
for idx in $(seq "$START_IDX" "$END_IDX"); do
    chunk_files+=("${json_files[idx]}")
done

echo "Processing ${#chunk_files[@]} files (indices $START_IDX-$END_IDX)..."

# Run the batch function: EternaFold + RibonanzaNet per-sample, then a single
# AF3 invocation for the whole chunk, then assemble output JSONs.
"${PYTHON_BIN}" "${NA_EVAL_UTILS}" \
    --function_name "process_designs_monomer_rna_batch" \
    --subject_paths "${chunk_files[@]}" \
    --overall_output_directory "$OUTPUT_DIR" \
    ${SKIP_ALPHAFOLD3:+--skip_alphafold3}

if [[ $? -ne 0 ]]; then
    echo "ERROR: Batch processing failed" >&2
    exit 1
fi
