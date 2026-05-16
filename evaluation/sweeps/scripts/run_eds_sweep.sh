#!/bin/bash
#SBATCH -p icelake
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=eds_sweep
#SBATCH --output=logs/eds_sweep_%A_%a.out
#SBATCH --error=logs/eds_sweep_%A_%a.err
#
# EDS sweep: fixed checkpoint × multiple K (step count) values × valid set.
# Mirrors run_checkpoint_sweep.sh but with K instead of checkpoint step.
# Trajectories are written to <SWEEP_DIR>/trajectories/K_<K>/<ctx>/<struct>.jsonl.
#
# Required env vars:
#   SWEEP_DIR  — e.g. evaluation/sweeps/eds_sweep
#   CKPT_PATH  — e.g. dfm_base/s_54019.pt  (single checkpoint)
#   SCHED_DIR  — directory containing eds_schedule_K{K}.json files (e.g. evaluation/entropy)
#
# Submit:
#   cd /home/mh2167/rds/hpc-work/NA-MPNN
#   export SWEEP_DIR=$PWD/evaluation/sweeps/eds_sweep \
#          CKPT_PATH=$PWD/dfm_base/s_54019.pt \
#          SCHED_DIR=$PWD/evaluation/entropy
#   # Choose K values to sweep (must have matching eds_schedule_K*.json):
#   echo "8 16 32 64 128" > $SWEEP_DIR/selected_ks.txt
#   sbatch --array=0-39 evaluation/sweeps/scripts/run_eds_sweep.sh

REPO_ROOT="${REPO_ROOT:-/home/voko/Documents/NA-MPNN}"

if [[ -z "$SWEEP_DIR" ]]; then echo "Error: SWEEP_DIR must be set" >&2; exit 1; fi
if [[ -z "$CKPT_PATH" ]]; then echo "Error: CKPT_PATH must be set" >&2; exit 1; fi
if [[ -z "$SCHED_DIR" ]]; then echo "Error: SCHED_DIR must be set" >&2; exit 1; fi

CSV_DIR="${CSV_DIR:-${REPO_ROOT}/evaluation/sweeps/scripts/valid_datasets}"
NA_EVAL_UTILS="${NA_EVAL_UTILS:-${REPO_ROOT}/evaluation/na_eval_utils.py}"
KS_FILE="${SWEEP_DIR}/selected_ks.txt"

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
METHOD=${METHOD:-na_mpnn}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
SAVE_TRAJECTORIES=${SAVE_TRAJECTORIES:-1}

if [[ ! -f "$KS_FILE" ]]; then
    echo "Missing $KS_FILE — list one or more K values (one per line or whitespace-separated)." >&2
    exit 1
fi
mapfile -t KS < <(tr ' ' '\n' < "$KS_FILE" | grep -v '^$')
if (( ${#KS[@]} == 0 )); then
    echo "No K values listed in $KS_FILE" >&2
    exit 1
fi

# Build flat (K, context, structure_path) work list
mapfile -t WORK < <(
    "${PYTHON_BIN}" "${REPO_ROOT}/evaluation/sweeps/scripts/build_eds_worklist.py" "$CSV_DIR" "${KS[@]}"
)

total=${#WORK[@]}
if (( total == 0 )); then
    echo "No work items built — check selected_ks.txt and per-context CSVs." >&2
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total )) && END_IDX=$(( total - 1 ))

echo "=== eds_sweep task=${TASK_ID}/${NUM_JOBS}  items=[${START_IDX}..${END_IDX}] of ${total} ===" >&2
echo "settings: NUM_SAMPLES=$NUM_SAMPLES TEMPERATURE=$TEMPERATURE CKPT_PATH=$CKPT_PATH" >&2

if (( START_IDX > END_IDX )); then
    echo "Empty chunk — nothing to do." >&2
    exit 0
fi

for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    _line="${WORK[idx]}"; K="${_line%%$'\t'*}"; _rest="${_line#*$'\t'}"; CTX="${_rest%%$'\t'*}"; STRUCT_PATH="${_rest#*$'\t'}"

    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "  [$idx] Skip: checkpoint not found: $CKPT_PATH" >&2
        continue
    fi

    SCHED_PATH="${SCHED_DIR}/eds_schedule_K${K}.json"
    if [[ ! -f "$SCHED_PATH" ]]; then
        echo "  [$idx] Skip: schedule not found: $SCHED_PATH" >&2
        continue
    fi

    OUT="${SWEEP_DIR}/outputs/K_${K}/${CTX}"
    struct_basename=$(basename "$STRUCT_PATH")
    struct_no_gz="${struct_basename%.gz}"
    struct_name="${struct_no_gz%.cif}"
    struct_name="${struct_name%.pdb}"
    design_json_dir="$OUT/$struct_name/design_json"

    if [[ "$SKIP_COMPLETED" == "1" && -d "$design_json_dir" && -n "$(compgen -G "$design_json_dir/*.json")" ]]; then
        echo "  [$idx] K=$K ctx=$CTX $struct_name — already done, skipping" >&2
        continue
    fi

    mkdir -p "$OUT"
    echo "  [$idx] K=$K ctx=$CTX $struct_name" >&2

    cmd=(
        "${PYTHON_BIN}" "${NA_EVAL_UTILS}"
        --function_name "design_nucleic_acid_sequence"
        --structure_path "$STRUCT_PATH"
        --overall_output_directory "$OUT"
        --num_samples "$NUM_SAMPLES"
        --method "$METHOD"
        --temperature "$TEMPERATURE"
        --na_mpnn_model_path "$CKPT_PATH"
        --model_mode "dfm"
        --dfm_schedule "eds"
        --eds_schedule_path "$SCHED_PATH"
    )
    if [[ "$SAVE_TRAJECTORIES" == "1" ]]; then
        TRAJ_DIR="${SWEEP_DIR}/trajectories/K_${K}/${CTX}"
        cmd+=( --trajectory_dir "$TRAJ_DIR" )
    fi
    "${cmd[@]}"

    recovery_json="$OUT/$struct_name/recovery.json"
    if compgen -G "$design_json_dir/*.json" > /dev/null 2>&1; then
        python3 - "$design_json_dir" "$recovery_json" <<'PYEOF'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
d = {o["name"]: o["tool_reported_sequence_recovery"] for p in src.glob("*.json") for o in [json.loads(p.read_text())]}
dst.write_text(json.dumps(d))
PYEOF
    fi
done

echo "=== done task=${TASK_ID}/${NUM_JOBS} ===" >&2
