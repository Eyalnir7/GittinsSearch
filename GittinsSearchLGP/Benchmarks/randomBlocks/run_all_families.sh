#!/bin/bash

# Run families with selectable model modes.
#
# Usage:
#   ./run_all_families.sh [modes]
#
# modes  Comma-separated list of modes to run: 1, 2, 3  (default: 1)
#
# Mode 1: experimentName=TunedGittinsFullRun,  modelsDir=tuned_scripted/
# Mode 2: experimentName=rb2bTrain,              modelsDir=rb2blocks_scripted/
# Mode 3: experimentName=rb2b3bTrain,            modelsDir=rb2blocks3blocks_scripted/
#
# Examples:
#   ./run_all_families.sh        # mode 1 only
#   ./run_all_families.sh 1,2,3  # all modes
#   ./run_all_families.sh 2,3    # modes 2 and 3

MODES_ARG="${1:-1}"
run_mode() { [[ ",$MODES_ARG," == *",$1,"* ]]; }

# ============================================================================
# Configuration
# ============================================================================

# Maps family key "low_high_goals_blocked" -> "numWaypoints numTaskPlans"
declare -A FAMILY_HYPERPARAMS=(
  ["2_2_2_2"]="75 3"
  ["3_3_3_2"]="150 3"
  ["4_4_4_1"]="30 10"
)

declare -a FAMILIES=(
  "2 2 2 2"
  "3 3 3 2"
  "4 4 4 1"
)

# Get PROJECT_ROOT from environment or use default
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi

LEARNING_BASE="$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile"

print_section() {
  echo ""
  echo "-------------------------------------------"
  echo "$1"
  echo "-------------------------------------------"
}

run_families() {
  local experimentName="$1"
  local modelsDir="$2"

  echo ""
  echo "=========================================="
  echo "MODE: experimentName=$experimentName"
  echo "      modelsDir=$modelsDir"
  echo "=========================================="
  echo ""

  for family in "${FAMILIES[@]}"; do
    read -r low high goals blocked <<< "$family"

    key="${low}_${high}_${goals}_${blocked}"
    params="${FAMILY_HYPERPARAMS[$key]}"
    numWaypoints=$(echo "$params" | awk '{print $1}')
    numTaskPlans=$(echo "$params" | awk '{print $2}')

    print_section "Family: Objects[$low,$high], Goals=$goals, Blocked=$blocked"

    ./x.exe -numObjLowerBound $low \
        -numObjUpperBound $high \
        -numGoalsUpperBound $goals \
        -numBlockedGoalsUpperBound $blocked \
        -experimentName "$experimentName" \
        -numIterations 100 \
        -GNN/modelsDir "$modelsDir" \
        -mode "run" \
        -solver GITTINS \
        -predictionType GNN \
        -dataPercentage 1.0 \
        -GITTINS/numWaypoints $numWaypoints \
        -numTaskPlans $numTaskPlans \
        -Bandit/beta 0.99999 \
        -useDatasizeSubdir false

    if [ $? -ne 0 ]; then
      echo "ERROR: Run failed for family $key!"
      exit 1
    fi
  done
}

# ============================================================================
# Mode 1
# ============================================================================
if run_mode 1; then
  run_families "TunedGittinsFullRun" "$LEARNING_BASE/tuned_scripted/"
fi

# ============================================================================
# Mode 2
# ============================================================================
if run_mode 2; then
  run_families "rb2bTrain" "$LEARNING_BASE/rb2blocks_scripted/"
fi

# ============================================================================
# Mode 3
# ============================================================================
if run_mode 3; then
  run_families "rb2b3bTrain" "$LEARNING_BASE/rb2blocks3blocks_scripted/"
fi