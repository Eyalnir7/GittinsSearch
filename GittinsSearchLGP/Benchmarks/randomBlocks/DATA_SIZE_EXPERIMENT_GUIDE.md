# Data Size Experiment - Usage Guide

## Overview

This file provides a clean, focused way to run your percentage experiments by calling it multiple times with different parameters. It has three modes:

1. **generate** - Generate and save configurations
2. **run** - Run GITTINS (or ELS) on saved configurations
3. **tune** - Run ELS hyperparameter tuning on saved configurations

## Directory Structure

The experiment automatically creates this structure:

```
dataSizeExperiment/
└── obj2_2_goals2_blocked2/          # One directory per config family
    ├── configs/                      # Generated configurations
    │   ├── metadata.txt
    │   ├── config_list.csv
    │   ├── config_0.g
    │   ├── lgp_0.lgp
    │   └── ...
    └── results/                      # Experiment results
        ├── GITTINS_50_5_p0.2_timestamp.STOP.dat
        ├── GITTINS_50_5_p0.4_timestamp.STOP.dat
        ├── GITTINS_50_5_p1.0_timestamp.STOP.dat
        ├── best_hyperparams_timestamp.txt
        └── tuning/                   # Tuning results
            ├── tuning_summary_aggregated.csv
            └── tune_*.STOP.dat files
```

## Step-by-Step Workflow

### Step 1: Generate Configurations (Once per family)

For each configuration family, generate the configurations ONCE:

```bash
# Family 1: Objects [2,2], Goals 2, Blocked 2
./dataSizeExperiment -f base.cfg \
  mode:generate \
  numObjLowerBound:2 \
  numObjUpperBound:2 \
  numGoalsUpperBound:2 \
  numBlockedGoalsUpperBound:2 \
  numIterations:100 \
  runSeed:0

# Family 2: Objects [4,4], Goals 4, Blocked 1  
./dataSizeExperiment -f base.cfg \
  mode:generate \
  numObjLowerBound:4 \
  numObjUpperBound:4 \
  numGoalsUpperBound:4 \
  numBlockedGoalsUpperBound:1 \
  numIterations:100 \
  runSeed:0

# Family 3: Objects [3,3], Goals 3, Blocked 2
./dataSizeExperiment -f base.cfg \
  mode:generate \
  numObjLowerBound:3 \
  numObjUpperBound:3 \
  numGoalsUpperBound:3 \
  numBlockedGoalsUpperBound:2 \
  numIterations:100 \
  runSeed:0
```

**What this does:**
- Creates 100 random configurations for each family
- Saves them to `dataSizeExperiment/obj{}_{}goals{}_blocked{}/configs/`
- Same configurations will be used for all subsequent experiments

---

### Step 2: Run GITTINS Experiments (Multiple data percentages)

For each configuration family, run with different data percentages:

```bash
# Family 1, Data 20%
./dataSizeExperiment -f gittins.cfg \
  mode:run \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.2 \
  numIterations:100 \
  runSeed:0

# Family 1, Data 40%
./dataSizeExperiment -f gittins.cfg \
  mode:run \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.4 \
  numIterations:100 \
  runSeed:0

# Family 1, Data 60%
./dataSizeExperiment -f gittins.cfg \
  mode:run \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.6 \
  numIterations:100 \
  runSeed:0

# ... repeat for 0.8, 1.0
# ... repeat for families 2 and 3
```

**What this does:**
- Loads the saved configurations
- Runs GITTINS solver with specified data percentage
- Saves results to `dataSizeExperiment/obj.../results/`

---

### Step 3: Run ELS Tuning (Once per family)

For each configuration family, run hyperparameter tuning:

```bash
# Family 1 - ELS Tuning
./dataSizeExperiment -f els_tuning.cfg \
  mode:tune \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.2 \
  numIterations:100 \
  runSeed:0

# Family 2 - ELS Tuning
./dataSizeExperiment -f els_tuning.cfg \
  mode:tune \
  numObjLowerBound:4 numObjUpperBound:4 \
  numGoalsUpperBound:4 numBlockedGoalsUpperBound:1 \
  dataPercentage:0.2 \
  numIterations:100 \
  runSeed:0

# Family 3 - ELS Tuning
./dataSizeExperiment -f els_tuning.cfg \
  mode:tune \
  numObjLowerBound:3 numObjUpperBound:3 \
  numGoalsUpperBound:3 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.2 \
  numIterations:100 \
  runSeed:0
```

**What this does:**
- Loads the saved configurations
- Tests all hyperparameter combinations
- Saves tuning results to `dataSizeExperiment/obj.../results/tuning/`
- Saves best parameters to `dataSizeExperiment/obj.../results/best_hyperparams_*.txt`

---

## Automation Scripts

### bash/run_all_experiments.sh

```bash
#!/bin/bash

# Configuration families
declare -a FAMILIES=(
  "2 2 2 2"
  "4 4 4 1"
  "3 3 3 2"
)

# Data percentages to test
PERCENTAGES=(0.2 0.4 0.6 0.8 1.0)

ITERATIONS=100
SEED=0

echo "=========================================="
echo "DATA SIZE EXPERIMENT - AUTOMATED RUN"
echo "=========================================="

# Step 1: Generate all configurations
echo ""
echo "STEP 1: Generating configurations..."
echo ""

for family in "${FAMILIES[@]}"; do
  read -r low high goals blocked <<< "$family"
  echo "→ Generating configs for Family: obj[$low,$high], goals$goals, blocked$blocked"
  
  ./dataSizeExperiment -f base.cfg \
    mode:generate \
    numObjLowerBound:$low \
    numObjUpperBound:$high \
    numGoalsUpperBound:$goals \
    numBlockedGoalsUpperBound:$blocked \
    numIterations:$ITERATIONS \
    runSeed:$SEED
done

# Step 2: Run GITTINS for all data percentages
echo ""
echo "STEP 2: Running GITTINS experiments..."
echo ""

for family in "${FAMILIES[@]}"; do
  read -r low high goals blocked <<< "$family"
  
  for perc in "${PERCENTAGES[@]}"; do
    echo "→ Running GITTINS: Family obj[$low,$high], Data $perc"
    
    ./dataSizeExperiment -f gittins.cfg \
      mode:run \
      numObjLowerBound:$low \
      numObjUpperBound:$high \
      numGoalsUpperBound:$goals \
      numBlockedGoalsUpperBound:$blocked \
      dataPercentage:$perc \
      numIterations:$ITERATIONS \
      runSeed:$SEED
  done
done

# Step 3: Run ELS tuning
echo ""
echo "STEP 3: Running ELS tuning..."
echo ""

for family in "${FAMILIES[@]}"; do
  read -r low high goals blocked <<< "$family"
  echo "→ Running ELS tuning: Family obj[$low,$high]"
  
  ./dataSizeExperiment -f els_tuning.cfg \
    mode:tune \
    numObjLowerBound:$low \
    numObjUpperBound:$high \
    numGoalsUpperBound:$goals \
    numBlockedGoalsUpperBound:$blocked \
    dataPercentage:0.2 \
    numIterations:$ITERATIONS \
    runSeed:$SEED
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
```

Make it executable:
```bash
chmod +x bash/run_all_experiments.sh
./bash/run_all_experiments.sh
```

---

## Manual Execution (For Testing)

### Test config generation only:

```bash
./dataSizeExperiment -f base.cfg mode:generate \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  numIterations:10 runSeed:0
```

Verify: Check that `dataSizeExperiment/obj2_2_goals2_blocked2/configs/` contains 10 config files.

### Test single GITTINS run:

```bash
./dataSizeExperiment -f gittins.cfg mode:run \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.2 numIterations:10 runSeed:0
```

Verify: Check that results appear in `dataSizeExperiment/obj2_2_goals2_blocked2/results/`

### Test tuning:

```bash
./dataSizeExperiment -f els_tuning.cfg mode:tune \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.2 numIterations:5 runSeed:0
```

Note: Use fewer iterations for testing tuning (it runs many combinations).

---

## Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `mode` | Operation mode: generate, run, or tune | `mode:run` |
| `numObjLowerBound` | Min objects in config family | `numObjLowerBound:2` |
| `numObjUpperBound` | Max objects in config family | `numObjUpperBound:5` |
| `numGoalsUpperBound` | Max goals | `numGoalsUpperBound:5` |
| `numBlockedGoalsUpperBound` | Max blocked goals | `numBlockedGoalsUpperBound:3` |
| `numIterations` | Number of configs/runs | `numIterations:100` |
| `dataPercentage` | GNN training data size | `dataPercentage:0.4` |
| `runSeed` | Random seed | `runSeed:0` |
| `solver` | Solver type (in cfg file) | `solver:GITTINS` or `solver:ELS` |



