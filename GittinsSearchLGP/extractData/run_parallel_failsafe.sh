#!/bin/bash

# ============================================================================
# Handle input parameters with validation
# ============================================================================

# Initialize variables
NUM_OBJ=""
NUM_GOALS=""
MAX_BLOCKED_GOALS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-objects)
            NUM_OBJ="$2"
            shift 2
            ;;
        --num-goals)
            NUM_GOALS="$2"
            shift 2
            ;;
        --max-blocked-goals)
            MAX_BLOCKED_GOALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num-objects NUM] [--num-goals NUM] [--max-blocked-goals NUM]"
            exit 1
            ;;
    esac
done

# Function to prompt user for input if not provided
prompt_for_input() {
    local var_name=$1
    local prompt_text=$2
    local value=""
    
    while [ -z "$value" ]; do
        read -p "$prompt_text" value
        if ! [[ "$value" =~ ^[0-9]+$ ]]; then
            echo "Error: Please enter a valid positive integer"
            value=""
        fi
    done
    echo "$value"
}

# Prompt for missing inputs
if [ -z "$NUM_OBJ" ]; then
    NUM_OBJ=$(prompt_for_input "NUM_OBJ" "Enter number of objects: ")
fi

if [ -z "$NUM_GOALS" ]; then
    NUM_GOALS=$(prompt_for_input "NUM_GOALS" "Enter number of goals: ")
fi

if [ -z "$MAX_BLOCKED_GOALS" ]; then
    MAX_BLOCKED_GOALS=$(prompt_for_input "MAX_BLOCKED_GOALS" "Enter maximum number of blocked goals: ")
fi

# Validate constraints
if ! [[ "$NUM_OBJ" =~ ^[0-9]+$ ]]; then
    echo "Error: Number of objects must be a positive integer"
    exit 1
fi

if ! [[ "$NUM_GOALS" =~ ^[0-9]+$ ]]; then
    echo "Error: Number of goals must be a positive integer"
    exit 1
fi

if ! [[ "$MAX_BLOCKED_GOALS" =~ ^[0-9]+$ ]]; then
    echo "Error: Maximum number of blocked goals must be a positive integer"
    exit 1
fi

# Validate logical constraints
if [ "$NUM_GOALS" -lt "$NUM_OBJ" ]; then
    echo "Error: Number of goals ($NUM_GOALS) must be greater than or equal to number of objects ($NUM_OBJ)"
    exit 1
fi

if [ "$MAX_BLOCKED_GOALS" -gt "$NUM_GOALS" ]; then
    echo "Error: Maximum number of blocked goals ($MAX_BLOCKED_GOALS) must be smaller than or equal to number of goals ($NUM_GOALS)"
    exit 1
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=".."
BASE_DATA_PATH="${PROJECT_ROOT}/data/reproduce_randomBlocks_${NUM_OBJ}objects_${NUM_GOALS}goals_${MAX_BLOCKED_GOALS}blockedgoals"
EXECUTABLE="${SCRIPT_DIR}/x.exe"
LOG_DIR="${SCRIPT_DIR}/logs/"
MAX_RETRIES=50  # Maximum number of restart attempts per agent

echo "[DEBUG] Creating logs directory at: $LOG_DIR"
mkdir -p "$LOG_DIR"
if [ -d "$LOG_DIR" ]; then
    echo "[DEBUG] Logs directory created successfully"
else
    echo "[ERROR] Failed to create logs directory at $LOG_DIR"
    exit 1
fi

echo "Configuration:"
echo "  Number of objects: $NUM_OBJ"
echo "  Number of goals: $NUM_GOALS"
echo "  Maximum blocked goals: $MAX_BLOCKED_GOALS"
echo "  Base data path: $BASE_DATA_PATH"
echo "  script directory: $SCRIPT_DIR"
echo "  executable path: $EXECUTABLE"
echo ""

# Function to get the last completed config ID for an agent
get_last_config_id() {
    local agent_id=$1
    local config_dir="${BASE_DATA_PATH}/agent_${agent_id}/configs"
    
    if [ ! -d "$config_dir" ]; then
        echo "-1"
        return
    fi
    
    # Find all z.conf*.g files and extract the highest number
    local max_id=$(find "$config_dir" -name "z.conf*.g" -type f | \
                   sed 's/.*z\.conf\([0-9]*\)\.g/\1/' | \
                   sort -n | tail -1)
    
    if [ -z "$max_id" ]; then
        echo "-1"
    else
        echo "$max_id"
    fi
}

# Function to run a single agent with restart capability
run_agent_with_restart() {
    local agent_id=$1
    local retry_count=0
    local start_config=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        echo "[Agent $agent_id] Starting run (attempt $((retry_count+1))/$MAX_RETRIES) from config $start_config at $(date)"
        
        # Run the agent (log only stderr)
        $EXECUTABLE --agent_id "$agent_id" --start_config_id "$start_config" --dataPath "$BASE_DATA_PATH" --numObjLowerBound "$NUM_OBJ" --numObjUpperBound "$NUM_OBJ" --numGoalsUpperBound "$NUM_GOALS" --numBlockedGoalsUpperBound "$MAX_BLOCKED_GOALS" --num_plans 3 --num_seed_trials 5 --num_problems 70 --num_waypoints_tries 500\
            2> "${LOG_DIR}/log${NUM_OBJ}${NUM_GOALS}${MAX_BLOCKED_GOALS}_${agent_id}_retry${retry_count}.txt" > /dev/null
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "[Agent $agent_id] Completed successfully at $(date)"
            return 0
        else
            echo "[Agent $agent_id] Failed with exit code $exit_code at $(date)"
            
            # Find where it crashed
            local last_completed=$(get_last_config_id "$agent_id")
            local next_config=$((last_completed + 1))
            
            echo "[Agent $agent_id] Last completed config: $last_completed"
            echo "[Agent $agent_id] Will restart from config: $next_config"
            
            # Check if we've made progress
            if [ "$next_config" -le "$start_config" ]; then
                echo "[Agent $agent_id] No progress made, waiting before retry..."
                sleep 5
            fi
            
            start_config=$next_config
            retry_count=$((retry_count + 1))
            
            if [ $retry_count -lt $MAX_RETRIES ]; then
                echo "[Agent $agent_id] Restarting in 2 seconds..."
                sleep 2
            fi
        fi
    done
    
    echo "[Agent $agent_id] Failed after $MAX_RETRIES attempts"
    return 1
}

# Trap SIGINT to cleanly stop all background processes
trap "echo 'Stopping all agents...'; kill 0; exit" SIGINT

# Array to store background PIDs
declare -a pids

# Start all agents in background
for i in {0..4}; do
    run_agent_with_restart $i &
    pids+=($!)
done

# Wait for all agents and track failures
failed_agents=()
for i in "${!pids[@]}"; do
    agent_id=$((i + 10))
    wait ${pids[$i]}
    if [ $? -ne 0 ]; then
        failed_agents+=($agent_id)
    fi
done

# Report final status
echo "========================================"
echo "All agents finished at $(date)"
if [ ${#failed_agents[@]} -eq 0 ]; then
    echo "All agents completed successfully!"
else
    echo "Failed agents: ${failed_agents[*]}"
    exit 1
fi
