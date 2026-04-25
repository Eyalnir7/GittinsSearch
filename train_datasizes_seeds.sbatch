#!/bin/bash
#SBATCH --job-name=my_grid
#SBATCH --array=0-449
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

# -------- Environment setup --------
cd "$SLURM_SUBMIT_DIR"
export APPTAINER_BIND="$HOME,$PRJ_WORKSPACE"
source .venv/bin/activate
cd GittinsSearchLGP/Learning

# -------- Parameter Definitions --------
NODE_TYPES=(waypoints rrt lgp)
TASKS=(feasibility feas_quantile infeas_quantile)
DATA_SIZES=(0.2 0.4 0.6 0.8 1.0)
SEEDS=({1..10})

NUM_TASKS=${#TASKS[@]}
NUM_NODES=${#NODE_TYPES[@]}
NUM_DATA=${#DATA_SIZES[@]}
NUM_SEEDS=${#SEEDS[@]}

# -------- Index Decoding (The "Nested Loop" Logic) --------
# Logic: ID = (((node * tasks) + task) * data_sizes + data) * seeds + seed
idx=$SLURM_ARRAY_TASK_ID

seed_i=$(( idx % NUM_SEEDS ))
idx=$(( idx / NUM_SEEDS ))

data_i=$(( idx % NUM_DATA ))
idx=$(( idx / NUM_DATA ))

task_i=$(( idx % NUM_TASKS ))
idx=$(( idx / NUM_TASKS ))

node_i=$(( idx % NUM_NODES ))

# Assign values
node_type="${NODE_TYPES[$node_i]}"
task="${TASKS[$task_i]}"
DATA_PCT="${DATA_SIZES[$data_i]}"
SEED="${SEEDS[$seed_i]}"

# -------- Skip forbidden combinations --------
if [[ "$node_type" == "rrt" && ( "$task" == "feasibility" || "$task" == "infeas_quantile" ) ]]; then
    echo "Skipping forbidden combination: $node_type + $task"
    exit 0
fi

echo "Job $SLURM_ARRAY_TASK_ID: node=$node_type, task=$task, data=$DATA_PCT, seed=$SEED"

python ./train_WANDB.py \
    --node-type "$node_type" \
    --task "$task" \
    --percentiles 0.1 0.3 0.5 0.7 0.9 \
    --data-percentage "$DATA_PCT" \
    --seed "$SEED" \
    --project "randomBlocksDataSizeTuned"