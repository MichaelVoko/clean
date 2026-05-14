#!/bin/bash
# Submit checkpoint sweep for new dfm_base steps (54511–58504, every other).
# Already appended to selected_steps.txt; completed structures are skipped.
# Run from repo root.

cd /rds/user/mh2167/hpc-work/NA-MPNN

export SWEEP_DIR=$PWD/evaluation/sweeps/dfm_sweep
export CKPT_DIR=$PWD/dfm_base
export MODEL_MODE=dfm
export DFM_DT=0.1

SAMPLES_CSV="$SWEEP_DIR/recovery_samples.csv"
[[ ! -f "$SAMPLES_CSV" ]] && echo "step,context,structure,sample,recovery" > "$SAMPLES_CSV"

sbatch --export=ALL --array=0-99 --ntasks=1 --cpus-per-task=1 --time=01:00:00 -p icelake  --mem=32g --job-name=check_sweep --error=logs/dt_sweep_%A_%a.err --output=logs/dt_sweep_%A_%a.out evaluation/sweeps/scripts/run_checkpoint_sweep.sh

# After completion:
#   python evaluation/sweeps/scripts/collect_recovery_v2.py
#   python evaluation/sweeps/scripts/plot_recovery.py
