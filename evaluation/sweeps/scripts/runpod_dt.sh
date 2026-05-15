cd /workspace/NA-MPNN

export REPO_ROOT=$PWD
export SWEEP_DIR=$PWD/evaluation/sweeps/dt_sweep_s27541
export CKPT=$PWD/dfm_model/s_27541.pt
export DT_VALUES="0.05 0.1 0.33 0.5 1.0"
export CSV_DIR=$PWD/evaluation/sweeps/scripts/valid_datasets
export NA_EVAL_UTILS=$PWD/evaluation/na_eval_utils.py
export NUM_SAMPLES=1
export CONDA_INIT=/workspace/miniconda3/etc/profile.d/conda.sh
export NA_EVAL_PYTHON_BIN=/workspace/miniconda3/envs/NA-MPNN/bin/python

# Initialize CSV header
[[ ! -f "$SWEEP_DIR/recovery_samples.csv" ]] && mkdir -p "$SWEEP_DIR" && echo "dt,context,structure,sample,recovery" > "$SWEEP_DIR/recovery_samples.csv"

# Run (serially, no SLURM)
SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 bash evaluation/sweeps/scripts/run_dt_sweep.sh

# Then collect and plot
/workspace/miniconda3/envs/NA-MPNN/bin/python evaluation/sweeps/scripts/collect_and_plot_dt.py \
  --sweep-dir "$SWEEP_DIR" \
  --out-csv "$SWEEP_DIR/recovery_by_dt.csv" \
  --out-png "$SWEEP_DIR/recovery_by_dt.png"
