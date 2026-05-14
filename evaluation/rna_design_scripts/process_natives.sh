#!/bin/bash

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NA_EVAL_UTILS="${NA_EVAL_UTILS:-${REPO_ROOT}/evaluation/na_eval_utils.py}"
PYTHON_BIN="${NA_EVAL_PYTHON_BIN:-python}"

CSV_FILE=$1
PROCESSED_REF_DIR=$2

if [ -z "$CSV_FILE" ] || [ -z "$PROCESSED_REF_DIR" ]; then
    echo "Usage: $0 <csv_file> <processed_reference_directory>"
    exit 1
fi

# 1) Load the "structure_path" column from the CSV into a temp file
TMP_PDB_PATHS="$(mktemp)"
trap 'rm -f "$TMP_PDB_PATHS"' EXIT

"${PYTHON_BIN}" - "$CSV_FILE" <<'PYCODE' > "$TMP_PDB_PATHS"
import sys, pandas as pd

df = pd.read_csv(sys.argv[1])

for p in df['structure_path']:
    print(p)
PYCODE

# 2) Loop over each path and invoke the Python processor
while IFS= read -r pdb_path; do
    [ -z "$pdb_path" ] && continue
    echo "$pdb_path"
    "${PYTHON_BIN}" "${NA_EVAL_UTILS}" \
        --function_name process_reference_monomer_rna \
        --reference_structure_path "$pdb_path" \
        --overall_output_directory "$PROCESSED_REF_DIR"
done < "$TMP_PDB_PATHS"
