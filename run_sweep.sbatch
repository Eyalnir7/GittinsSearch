#!/bin/bash
#SBATCH --job-name=my_grid
#SBATCH --array=0-150
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

usage() {
    echo "Usage: $0 <project_name> <sweep_id> <node_type> <task> <data_percentage>" >&2
    exit 1
}

if [ "$#" -ne 5 ]; then
    usage
fi

PROJECT_NAME="$1"
SWEEP_ID="$2"
NODE_TYPE="$3"
TASK="$4"
DATA_PERCENTAGE="$5"

LOG_DIR="logs-sweep-${NODE_TYPE}-${TASK}"
mkdir -p "$LOG_DIR"
LOG_PREFIX="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
exec > "${LOG_PREFIX}.out" 2> "${LOG_PREFIX}.err"

# Go to repo root (where you ran sbatch)
cd "$SLURM_SUBMIT_DIR"

export APPTAINER_BIND="$HOME,$PRJ_WORKSPACE"
source .venv/bin/activate

# Move into the training directory
cd GittinsSearchLGP/Learning

echo "Running in $(pwd)"

python ./train_WANDB.py \
    --node-type "$NODE_TYPE" \
    --task "$TASK" \
    --percentiles 0.1 0.3 0.5 0.7 0.9 \
    --datadir ../data/randomBlocks_all/ \
    --data-percentage "$DATA_PERCENTAGE" \
    --project "$PROJECT_NAME" \
    sweep 1 "$SWEEP_ID"
