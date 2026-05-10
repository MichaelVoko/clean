#!/bin/bash
#SBATCH -p ampere
#SBATCH -A LIO-SL3-GPU
#SBATCH --job-name=train
#SBATCH --output=train_log/train_%j.out
#SBATCH --error=train_log/train_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32g
#SBATCH -c 20
#SBATCH -t 01:00:00
#SBATCH --signal=B:USR1@120
_JSON="/home/stu/clean/dfm_model.json"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NA-MPNN

export WANDB_MODE=offline
export WANDB_DIR=/home/stu/clean/logs/wandb

# Persist a single wandb run id across resubmits so all offline runs
# stitch into one continuous run when later synced from a login node.
WANDB_ID_FILE=/home/stu/clean/dfm_base/wandb_run.id
mkdir -p "$(dirname "$WANDB_ID_FILE")"
if [ ! -s "$WANDB_ID_FILE" ]; then
    python -c "import wandb; print(wandb.util.generate_id())" > "$WANDB_ID_FILE"
fi
export WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
export WANDB_RESUME=allow

# Resubmit handler — triggered 120s before time limit
resubmit() {
    echo "Received signal → resubmitting before time limit..."
    sbatch "$(realpath "$0")"
    # Let torchrun save its checkpoint before SLURM kills us
    wait
    exit 0
}
trap resubmit USR1

echo "Starting job $SLURM_JOB_ID at $(date)"

# Run in background so the trap can fire
torchrun --nproc_per_node=4 /home/stu/clean/na_run.py "$_JSON" &
TRAIN_PID=$!
wait $TRAIN_PID
EXIT_CODE=$?

echo "Job finished with exit code $EXIT_CODE at $(date)"

# Resubmit unless training signaled completion (exit code 0)
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "Non-zero exit → resubmitting..."
    sbatch "$(realpath "$0")"
else
    echo "Training complete → not resubmitting."
fi