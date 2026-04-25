#!/bin/bash

# Data Size Experiment - Automated Runner
# This script runs the complete experiment pipeline:
# 1. Generate configurations for each family
# 2. Run GITTINS with multiple data percentages
# 3. Run ELS hyperparameter tuning
# 4. Run Gittins hyperparameter tuning
#
# Usage:
#   ./run_all_experiments.sh [steps]
#
# steps  Comma-separated list of steps to run: 1, 2, 3, 4  (default: 1,2,3,4)
#
# Examples:
#   ./run_all_experiments.sh          # run all steps
#   ./run_all_experiments.sh 2        # GITTINS only
#   ./run_all_experiments.sh 1,3      # generate + ELS tuning
#   ./run_all_experiments.sh 2,3      # GITTINS + ELS tuning
#   ./run_all_experiments.sh 4        # Gittins tuning only
#   ./run_all_experiments.sh 1,4      # generate + Gittins tuning

# Parse step selection (default: all)
STEPS_ARG="${1:-1,2,3,4}"
run_step() { [[ ",$STEPS_ARG," == *",$1,"* ]]; }

# ============================================================================
# Configuration
# ============================================================================

# Get PROJECT_ROOT from environment or use default
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi

# Configuration families: "lowObj highObj maxGoals maxBlocked"
declare -a FAMILIES=(
  "2 2 2 2"
  "3 3 3 2"
  "4 4 4 1"
)

# Maps family key "low_high_goals_blocked" -> "numWaypoints numTaskPlans"
declare -A FAMILY_HYPERPARAMS=(
  ["2_2_2_2"]="75 3"
  ["3_3_3_2"]="150 3"
  ["4_4_4_1"]="30 10"
)

# Data percentages to test with GITTINS
# PERCENTAGES=(0.2 0.4 0.6 0.8 1.0)
PERCENTAGES=(0.6 1.0)

# Model seeds to test with GITTINS (subdirectories seed_1 through seed_10)
MODEL_SEEDS=(1 2 3 4 5 6 7 8 9 10)

# Experiment parameters
ITERATIONS=20
SEED=0

# Executable name (adjust if needed)
EXECUTABLE="./x.exe"

# Config files
CONFIG_BASE="config_base.cfg"
CONFIG_GITTINS="config_gittins.cfg"
CONFIG_ELS="config_els_tuning.cfg"

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
  echo ""
  echo "=========================================="
  echo "$1"
  echo "=========================================="
  echo ""
}

print_section() {
  echo ""
  echo "-------------------------------------------"
  echo "$1"
  echo "-------------------------------------------"
}

# ============================================================================
# Main Script
# ============================================================================

print_banner "DATA SIZE EXPERIMENT - AUTOMATED RUN"

echo "Configuration:"
echo "  Steps to run: $STEPS_ARG"
echo "  Families: ${#FAMILIES[@]}"
echo "  Data Percentages: ${PERCENTAGES[@]}"
echo "  Model Seeds: ${MODEL_SEEDS[@]}"
echo "  Iterations per run: $ITERATIONS"
echo "  Random Seed: $SEED"
echo ""

# Ask for confirmation
read -p "Start experiment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# STEP 1: Generate Configurations
# ============================================================================

if run_step 1; then

print_banner "STEP 1: GENERATING CONFIGURATIONS"

for family in "${FAMILIES[@]}"; do
  read -r low high goals blocked <<< "$family"
  
  print_section "Family: Objects[$low,$high], Goals=$goals, Blocked=$blocked"
  
  $EXECUTABLE \
    -mode generate \
    -numObjLowerBound $low \
    -numObjUpperBound $high \
    -numGoalsUpperBound $goals \
    -numBlockedGoalsUpperBound $blocked \
    -numIterations $ITERATIONS \
    -runSeed $SEED \
    -solver GITTINS
  
  if [ $? -ne 0 ]; then
    echo "ERROR: Configuration generation failed!"
    exit 1
  fi
done

print_banner "✓ CONFIGURATION GENERATION COMPLETE"

fi # end step 1

# ============================================================================
# STEP 2: Run GITTINS Experiments
# ============================================================================

if run_step 2; then

print_banner "STEP 2: RUNNING GITTINS EXPERIMENTS"

total_gittins_runs=$((${#FAMILIES[@]} * ${#PERCENTAGES[@]} * ${#MODEL_SEEDS[@]}))
current_run=0

for family in "${FAMILIES[@]}"; do
  read -r low high goals blocked <<< "$family"
  
  for perc in "${PERCENTAGES[@]}"; do
    for model_seed in "${MODEL_SEEDS[@]}"; do
      current_run=$((current_run + 1))
      
      print_section "Run $current_run/$total_gittins_runs: Objects[$low,$high], Data=$perc, ModelSeed=$model_seed"
      
      key="${low}_${high}_${goals}_${blocked}"
      params="${FAMILY_HYPERPARAMS[$key]}"
      numWaypoints=$(echo "$params" | awk '{print $1}')
      numTaskPlans=$(echo "$params" | awk '{print $2}')

      $EXECUTABLE \
        -mode run \
        -numObjLowerBound $low \
        -numObjUpperBound $high \
        -numGoalsUpperBound $goals \
        -numBlockedGoalsUpperBound $blocked \
        -dataPercentage $perc \
        -numIterations $ITERATIONS \
        -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/dataSizeScripted/" \
        -runSeed $SEED \
        -modelSeed $model_seed \
        -useDatasizeSubdir true \
        -Bandit/beta 0.99999 \
        -solver GITTINS \
        -GITTINS/numWaypoints $numWaypoints \
        -numTaskPlans $numTaskPlans
      
      if [ $? -ne 0 ]; then
        echo "WARNING: GITTINS run failed, continuing..."
      fi
    done
  done
done

print_banner "✓ GITTINS EXPERIMENTS COMPLETE"

fi # end step 2

# ============================================================================
# STEP 3: Run ELS Tuning
# ============================================================================

if run_step 3; then

print_banner "STEP 3: RUNNING ELS HYPERPARAMETER TUNING"

tuning_count=0
for family in "${FAMILIES[@]}"; do
  read -r low high goals blocked <<< "$family"
  tuning_count=$((tuning_count + 1))
  
  print_section "Tuning $tuning_count/${#FAMILIES[@]}: Objects[$low,$high]"
  
  $EXECUTABLE \
    -mode tune \
    -numObjLowerBound $low \
    -numObjUpperBound $high \
    -numGoalsUpperBound $goals \
    -numBlockedGoalsUpperBound $blocked \
    -dataPercentage 0.2 \
    -numIterations $ITERATIONS \
    -runSeed $SEED \
    -solver ELS
  
  if [ $? -ne 0 ]; then
    echo "WARNING: ELS tuning failed, continuing..."
  fi
done

print_banner "✓ ELS TUNING COMPLETE"

fi # end step 3

# ============================================================================
# STEP 4: Run Gittins Hyperparameter Tuning
# ============================================================================

if run_step 4; then

print_banner "STEP 4: RUNNING GITTINS HYPERPARAMETER TUNING"

gittins_tuning_count=0
for family in "${FAMILIES[@]}"; do
  read -r low high goals blocked <<< "$family"
  gittins_tuning_count=$((gittins_tuning_count + 1))

  print_section "Tuning $gittins_tuning_count/${#FAMILIES[@]}: Objects[$low,$high]"

  $EXECUTABLE \
    -mode tuneGittins \
    -experimentName "TuningGittins" \
    -numObjLowerBound $low \
    -numObjUpperBound $high \
    -numGoalsUpperBound $goals \
    -numBlockedGoalsUpperBound $blocked \
    -numIterations $ITERATIONS \
    -runSeed $SEED \
    -solver GITTINS \
    -GNN/modelsDir "$PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/tuned_scripted/" \
    -useDatasizeSubdir false


  if [ $? -ne 0 ]; then
    echo "WARNING: Gittins tuning failed, continuing..."
  fi
done

print_banner "✓ GITTINS TUNING COMPLETE"

fi # end step 4

# ============================================================================
# Summary
# ============================================================================

print_banner "ALL EXPERIMENTS COMPLETE!"

echo "Results saved in: dataSizeExperiment/"
echo ""
echo "Summary:"
echo "  - ${#FAMILIES[@]} configuration families generated"
echo "  - $total_gittins_runs GITTINS runs completed (${#PERCENTAGES[@]} percentages × ${#MODEL_SEEDS[@]} model seeds × ${#FAMILIES[@]} families)"
echo "  - ${#FAMILIES[@]} ELS tuning sessions completed"
echo "  - ${#FAMILIES[@]} Gittins tuning sessions completed"
echo ""
echo "Next steps:"
echo "  1. Check results in dataSizeExperiment/obj*/results/"
echo "  2. Review ELS tuning results in dataSizeExperiment/obj*/results/tuning/"
echo "  3. Review Gittins tuning results in dataSizeExperiment/obj*/results/*/tuning_gittins/"
echo "  4. Analyze best_hyperparams_*.txt and best_gittins_hyperparams_*.txt files"
echo ""
