#!/bin/bash
#SBATCH --job-name=my_grid
#SBATCH --array=0-8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH --time=12:00:00
#SBATCH --output=logs1/%x-%A_%a.out
#SBATCH --error=logs1/%x-%A_%a.err

set -euo pipefail
mkdir -p logs1

# -------- Project Name Setup --------
# Use the first command line argument, or default to 'randomBlocksSweeps'
PROJECT_NAME="${1:-randomBlocksSweeps}"
TRAIN_ARTIFACT="${2:-randomBlocks_all}"

# -------- Environment setup --------
cd "$SLURM_SUBMIT_DIR"

export APPTAINER_BIND="$HOME,$PRJ_WORKSPACE"
source .venv/bin/activate

cd GittinsSearchLGP/Learning

# -------- Grid definition --------
NODE_TYPES=(waypoints rrt lgp)
TASKS=(feasibility feas_quantile infeas_quantile)

NUM_TASKS=${#TASKS[@]}

# Decode array index (node_type × task)
node_i=$(( SLURM_ARRAY_TASK_ID / NUM_TASKS ))
task_i=$(( SLURM_ARRAY_TASK_ID % NUM_TASKS ))

node_type="${NODE_TYPES[$node_i]}"
task="${TASKS[$task_i]}"

# -------- Skip forbidden combinations --------
if [[ "$node_type" == "rrt" && ( "$task" == "feasibility" || "$task" == "infeas_quantile" ) ]]; then
    echo "Skipping forbidden combination: $node_type + $task"
    exit 0
fi

echo "Running in $(pwd)"
echo "Project Name: $PROJECT_NAME"
echo "node-type=$node_type task=$task array_id=$SLURM_ARRAY_TASK_ID"

python ./train_WANDB.py \
    --node-type "$node_type" \
    --task "$task" \
    --percentiles 0.1 0.3 0.5 0.7 0.9 \
    --project "$PROJECT_NAME" \
    --train-artifact-name "$TRAIN_ARTIFACT" 

