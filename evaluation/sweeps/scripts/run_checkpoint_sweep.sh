#!/bin/bash
#SBATCH -p icelake
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=ar_sweep
#SBATCH --output=logs/dfm_sweep_%A_%a.out
#SBATCH --error=logs/dfm_sweep_%A_%a.err
#
# Sweep evaluation — mirrors design_sequences.sh's sharding pattern.
# The full (checkpoint, context, structure) work list is flattened, then
# chunked across SLURM_ARRAY_TASK_COUNT tasks. Pick any --array=0-N-1 to
# run N parallel tasks: larger N  ⇒  smaller chunks  ⇒  more concurrency.
# Resume: structures whose design_json/ dir already contains *.json are
# skipped (SKIP_COMPLETED=1, same semantics as design_sequences.sh).
#
# Required env vars (no defaults — must be set explicitly):
#   SWEEP_DIR  — directory for selected_steps.txt and outputs/ (e.g. evaluation/sweeps/dfm_sweep)
#   CKPT_DIR   — directory containing s_<step>.pt checkpoints (e.g. dfm_base or ar_model)
#
# Submit for DFM mode:
#   cd /home/mh2167/rds/hpc-work/NA-MPNN
#   export SWEEP_DIR=$PWD/evaluation/sweeps/dfm_sweep CKPT_DIR=$PWD/dfm_base MODEL_MODE=dfm
#   python evaluation/sweeps/scripts/select_checkpoints.py --ckpt-dir $CKPT_DIR --out $SWEEP_DIR/selected_steps.txt --n 25
#   sbatch --array=0-199 evaluation/sweeps/scripts/run_checkpoint_sweep.sh
#
# Submit for AR mode:
#   cd /home/mh2167/rds/hpc-work/NA-MPNN
#   export SWEEP_DIR=$PWD/evaluation/sweeps/ar_sweep CKPT_DIR=$PWD/ar_model MODEL_MODE=ar
#   python evaluation/sweeps/scripts/select_checkpoints.py --ckpt-dir $CKPT_DIR --out $SWEEP_DIR/selected_steps.txt --n 25
#   sbatch --array=0-199 evaluation/sweeps/scripts/run_checkpoint_sweep.sh
#
# CSV_DIR defaults to evaluation/sweeps/scripts/valid_datasets (both modes use the same validation splits).
# To resume a partial run, just re-submit — completed structures are skipped automatically.

REPO_ROOT="${REPO_ROOT:-/home/voko/Documents/NA-MPNN}"

if [[ -z "$SWEEP_DIR" ]]; then
    echo "Error: SWEEP_DIR must be set (e.g. \$PWD/evaluation/sweeps/dfm_sweep or \$PWD/evaluation/sweeps/ar_sweep)" >&2
    exit 1
fi
if [[ -z "$CKPT_DIR" ]]; then
    echo "Error: CKPT_DIR must be set (e.g. \$PWD/dfm_base or \$PWD/ar_model)" >&2
    exit 1
fi
CSV_DIR="${CSV_DIR:-${REPO_ROOT}/evaluation/sweeps/scripts/valid_datasets}"
NA_EVAL_UTILS="${NA_EVAL_UTILS:-${REPO_ROOT}/evaluation/na_eval_utils.py}"
STEPS_FILE="${SWEEP_DIR}/selected_steps.txt"

CONDA_ROOT="${CONDA_ROOT:-/home/voko/miniconda3}"
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-NA-MPNN}"
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || \
        echo "Warning: conda env '$CONDA_ENV_NAME' not found; using current Python." >&2
fi
PYTHON_BIN="${NA_EVAL_PYTHON_BIN:-$(command -v python)}"

NUM_SAMPLES=${NUM_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.1}
MODEL_MODE=${MODEL_MODE:-dfm}
DFM_DT=${DFM_DT:-0.1}
METHOD=${METHOD:-na_mpnn}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
CONTEXTS=(rna_only rna_with_protein dna_only dna_with_protein)

# 1) sanity-check prep files exist
if [[ ! -f "$STEPS_FILE" ]]; then
    echo "Missing $STEPS_FILE — run select_checkpoints.py first." >&2
    exit 1
fi
mapfile -t STEPS < "$STEPS_FILE"
if (( ${#STEPS[@]} == 0 )); then
    echo "No steps listed in $STEPS_FILE" >&2
    exit 1
fi

# 2) build flat (step, context, structure_path) work list
mapfile -t WORK < <(
    "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/sweeps/scripts/build_worklist.py" "$CSV_DIR" "${STEPS[@]}"
)

total=${#WORK[@]}
if (( total == 0 )); then
    echo "No work items built — check steps file and per-context CSVs." >&2
    exit 1
fi

# 3) compute chunking for this array task
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total )) && END_IDX=$(( total - 1 ))

echo "=== dfm_sweep task=${TASK_ID}/${NUM_JOBS}  items=[${START_IDX}..${END_IDX}] of ${total} ===" >&2
echo "settings: NUM_SAMPLES=$NUM_SAMPLES TEMPERATURE=$TEMPERATURE MODEL_MODE=$MODEL_MODE DFM_DT=$DFM_DT" >&2

if (( START_IDX > END_IDX )); then
    echo "Empty chunk — nothing to do." >&2
    exit 0
fi

# 4) process this shard
for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    _line="${WORK[idx]}"; STEP="${_line%%$'\t'*}"; _rest="${_line#*$'\t'}"; CTX="${_rest%%$'\t'*}"; STRUCT_PATH="${_rest#*$'\t'}"

    CKPT="${CKPT_DIR}/s_${STEP}.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "  [$idx] Skip: checkpoint not found: $CKPT" >&2
        continue
    fi

    OUT="${SWEEP_DIR}/outputs/step_${STEP}/${CTX}"
    struct_basename=$(basename "$STRUCT_PATH")
    struct_no_gz="${struct_basename%.gz}"
    struct_name="${struct_no_gz%.cif}"
    struct_name="${struct_name%.pdb}"
    design_json_dir="$OUT/$struct_name/design_json"

    if [[ "$SKIP_COMPLETED" == "1" && -d "$design_json_dir" && -n "$(compgen -G "$design_json_dir/*.json")" ]]; then
        echo "  [$idx] step=$STEP ctx=$CTX $struct_name — already done, skipping" >&2
        continue
    fi

    mkdir -p "$OUT"
    echo "  [$idx] step=$STEP ctx=$CTX $struct_name" >&2

    cmd=(
        "${PYTHON_BIN}" "${NA_EVAL_UTILS}"
        --function_name "design_nucleic_acid_sequence"
        --structure_path "$STRUCT_PATH"
        --overall_output_directory "$OUT"
        --num_samples "$NUM_SAMPLES"
        --method "$METHOD"
        --temperature "$TEMPERATURE"
        --na_mpnn_model_path "$CKPT"
        --model_mode "$MODEL_MODE"
        --dfm_dt "$DFM_DT"
    )
    "${cmd[@]}"

    if compgen -G "$design_json_dir/*.json" > /dev/null 2>&1; then
        (
          flock -x 9
          python3 - "$design_json_dir" "$STEP" "$CTX" "$struct_name" \
              >> "$SWEEP_DIR/recovery_samples.csv" <<'PYEOF'
import json, sys
from pathlib import Path
src, step, ctx, struct = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
for p in sorted(src.glob("*.json")):
    d = json.loads(p.read_text())
    print(f"{step},{ctx},{struct},{p.stem},{d['tool_reported_sequence_recovery']}")
PYEOF
        ) 9>"$SWEEP_DIR/.recovery_samples.lock"
    fi
done

echo "=== done task=${TASK_ID}/${NUM_JOBS} ===" >&2
