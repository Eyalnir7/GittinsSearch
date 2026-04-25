#!/bin/bash
#SBATCH --job-name=my_grid
#SBATCH --array=0-44
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

# Go to repo root (where you ran sbatch)
cd "$SLURM_SUBMIT_DIR"

export APPTAINER_BIND="$HOME,$PRJ_WORKSPACE"
source .venv/bin/activate

# Move into the training directory
cd GittinsSearchLGP/Learning

NODE_TYPES=(waypoints rrt lgp)
TASKS=(feasibility feas_quantile infeas_quantile)
DATA_PCTS=(0.2 0.4 0.6 0.8 1)

NUM_TASKS=${#TASKS[@]}
NUM_PCTS=${#DATA_PCTS[@]}

# Decode array index
node_i=$(( SLURM_ARRAY_TASK_ID / (NUM_TASKS * NUM_PCTS) ))
rem=$(( SLURM_ARRAY_TASK_ID % (NUM_TASKS * NUM_PCTS) ))
task_i=$(( rem / NUM_PCTS ))
pct_i=$(( rem % NUM_PCTS ))

node_type="${NODE_TYPES[$node_i]}"
task="${TASKS[$task_i]}"
data_pct="${DATA_PCTS[$pct_i]}"

# Skip forbidden combinations
if [[ "$node_type" == "rrt" && ( "$task" == "feasibility" || "$task" == "infeas_quantile" ) ]]; then
    echo "Skipping forbidden combination: $node_type + $task"
    exit 0
fi

echo "Running in $(pwd)"
echo "node-type=$node_type task=$task data-percentage=$data_pct array_id=$SLURM_ARRAY_TASK_ID"

python ./train_WANDB.py \
    --no-wandb \
    --node-type "$node_type" \
    --task "$task" \
    --datadir ../data/randomBlocks_all/ \
    --percentiles 0.1 0.3 0.5 0.7 0.9 \
    --data-percentage "$data_pct"
