#!/bin/bash
#SBATCH -p icelake
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=dt_sweep
#SBATCH --output=logs/dt_sweep_%A_%a.out
#SBATCH --error=logs/dt_sweep_%A_%a.err
#
# DFM dt sweep — fixed checkpoint, variable Euler step size.
#
# Required env vars:
#   SWEEP_DIR  — output dir (e.g. evaluation/sweeps/dt_sweep)
#   CKPT       — path to a single checkpoint .pt file
#   DT_VALUES  — space-separated dt values (e.g. "0.05 0.025")
#
# Submit:
#   cd /home/mh2167/rds/hpc-work/NA-MPNN
#   export SWEEP_DIR=$PWD/evaluation/sweeps/dt_sweep
#   export CKPT=$PWD/dfm_base/s_XXXXX.pt
#   export DT_VALUES="0.05 0.025"
#   sbatch --array=0-99 evaluation/sweeps/scripts/run_dt_sweep.sh

REPO_ROOT="${REPO_ROOT:-/home/voko/Documents/NA-MPNN}"
CSV_DIR="${CSV_DIR:-${REPO_ROOT}/evaluation/sweeps/scripts/valid_datasets}"
NA_EVAL_UTILS="${NA_EVAL_UTILS:-${REPO_ROOT}/evaluation/na_eval_utils.py}"

[[ -z "$SWEEP_DIR" ]] && { echo "Error: SWEEP_DIR not set" >&2; exit 1; }
[[ -z "$CKPT"      ]] && { echo "Error: CKPT not set" >&2; exit 1; }
[[ -z "$DT_VALUES" ]] && { echo "Error: DT_VALUES not set" >&2; exit 1; }
[[ ! -f "$CKPT"    ]] && { echo "Error: checkpoint not found: $CKPT" >&2; exit 1; }

CONDA_INIT="${CONDA_INIT:-/home/voko/miniconda3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_INIT" ]] && source "$CONDA_INIT" && conda activate NA-MPNN 2>/dev/null
PYTHON_BIN="${NA_EVAL_PYTHON_BIN:-$(command -v python)}"

NUM_SAMPLES=${NUM_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

DT_ARRAY=($DT_VALUES)

mapfile -t WORK < <(
    "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/sweeps/scripts/build_worklist.py" "$CSV_DIR" "${DT_ARRAY[@]}"
)

total=${#WORK[@]}
(( total == 0 )) && { echo "No work items." >&2; exit 1; }

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total )) && END_IDX=$(( total - 1 ))
(( START_IDX > END_IDX )) && { echo "Empty chunk." >&2; exit 0; }

echo "=== dt_sweep task=${TASK_ID}/${NUM_JOBS} items=[${START_IDX}..${END_IDX}] of ${total} CKPT=$(basename $CKPT) DT_VALUES=$DT_VALUES ===" >&2

for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    _line="${WORK[idx]}"; DT="${_line%%$'\t'*}"; _rest="${_line#*$'\t'}"; CTX="${_rest%%$'\t'*}"; STRUCT_PATH="${_rest#*$'\t'}"
    OUT="${SWEEP_DIR}/dt_${DT}/${CTX}"
    struct_name=$(basename "${STRUCT_PATH%.gz}"); struct_name="${struct_name%.cif}"; struct_name="${struct_name%.pdb}"
    design_json_dir="$OUT/$struct_name/design_json"

    if [[ "$SKIP_COMPLETED" == "1" && -d "$design_json_dir" && -n "$(compgen -G "$design_json_dir/*.json")" ]]; then
        echo "  [$idx] dt=$DT $struct_name — skip" >&2; continue
    fi

    mkdir -p "$OUT"
    echo "  [$idx] dt=$DT ctx=$CTX $struct_name" >&2
    "${PYTHON_BIN}" "${NA_EVAL_UTILS}" \
        --function_name design_nucleic_acid_sequence \
        --structure_path "$STRUCT_PATH" \
        --overall_output_directory "$OUT" \
        --num_samples "$NUM_SAMPLES" \
        --method na_mpnn \
        --temperature "$TEMPERATURE" \
        --na_mpnn_model_path "$CKPT" \
        --model_mode dfm \
        --dfm_dt "$DT"

    if compgen -G "$design_json_dir/*.json" > /dev/null 2>&1; then
        (
          flock -x 9
          python3 - "$design_json_dir" "$DT" "$CTX" "$struct_name" \
              >> "$SWEEP_DIR/recovery_samples.csv" <<'PYEOF'
import json, sys
from pathlib import Path
src, dt, ctx, struct = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
for p in sorted(src.glob("*.json")):
    d = json.loads(p.read_text())
    print(f"{dt},{ctx},{struct},{p.stem},{d['tool_reported_sequence_recovery']}")
PYEOF
        ) 9>"$SWEEP_DIR/.recovery_samples.lock"
    fi
done

echo "=== done task=${TASK_ID}/${NUM_JOBS} ===" >&2
