#!/bin/bash
# Submit DFM dt sweep for dt=0.2, 0.33, 0.5 using checkpoint s_54019.pt.
# Results land in evaluation/sweeps/dt_sweep/dt_<value>/ alongside existing dt_0.025 and dt_0.05.
# Run from repo root.

cd /rds/user/mh2167/hpc-work/NA-MPNN

export SWEEP_DIR=$PWD/evaluation/sweeps/dt_sweep
export CKPT=$PWD/dfm_base/s_54019.pt
export DT_VALUES="0.2 0.33 0.5"

SAMPLES_CSV="$SWEEP_DIR/recovery_samples.csv"
[[ ! -f "$SAMPLES_CSV" ]] && echo "dt,context,structure,sample,recovery" > "$SAMPLES_CSV"

sbatch --export=ALL --array=0-29 --ntasks=1 --cpus-per-task=1 --time=01:00:00 -p icelake  --mem=32g --job-name=dt_sweep --error=logs/dt_sweep_%A_%a.err --output=logs/dt_sweep_%A_%a.out evaluation/sweeps/scripts/run_dt_sweep.sh

# After completion:
#   python evaluation/sweeps/scripts/collect_and_plot_dt.py
