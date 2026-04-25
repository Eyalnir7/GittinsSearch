# Data Size Experiment - Quick Reference Card

## Three Modes

```bash
mode:generate    # Generate and save configurations
mode:run         # Run experiments on saved configs  
mode:tune        # Run hyperparameter tuning
```

## Essential Commands

### 1. Generate Configs (Run Once)
```bash
./dataSizeExperiment -f config_base.cfg \
  mode:generate \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  numIterations:100 runSeed:0
```

### 2. Run GITTINS with Specific Data %
```bash
./dataSizeExperiment -f config_gittins.cfg \
  mode:run \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.4 \
  numIterations:100 runSeed:0
```

### 3. Run ELS Tuning
```bash
./dataSizeExperiment -f config_els_tuning.cfg \
  mode:tune \
  numObjLowerBound:2 numObjUpperBound:2 \
  numGoalsUpperBound:2 numBlockedGoalsUpperBound:2 \
  dataPercentage:0.2 \
  numIterations:100 runSeed:0
```

## Your Three Config Families

### Family 1: 2 objects, 2 goals, 2 blocked
```bash
numObjLowerBound:2 numObjUpperBound:2 numGoalsUpperBound:2 numBlockedGoalsUpperBound:2
```

### Family 2: 4 objects, 4 goals, 1 blocked  
```bash
numObjLowerBound:4 numObjUpperBound:4 numGoalsUpperBound:4 numBlockedGoalsUpperBound:1
```

### Family 3: 3 objects, 3 goals, 2 blocked
```bash
numObjLowerBound:3 numObjUpperBound:3 numGoalsUpperBound:3 numBlockedGoalsUpperBound:2
```

## Data Percentages to Test

```bash
dataPercentage:0.2
dataPercentage:0.4
dataPercentage:0.6
dataPercentage:0.8
dataPercentage:1.0
```

## Run Everything (Automated)

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

## Results Location

```
dataSizeExperiment/
├── obj2_2_goals2_blocked2/
│   ├── configs/              # Generated configs
│   └── results/              # All results here
│       ├── GITTINS_*_p0.2*.STOP.dat
│       ├── GITTINS_*_p0.4*.STOP.dat
│       ├── GITTINS_*_p0.6*.STOP.dat
│       ├── GITTINS_*_p0.8*.STOP.dat
│       ├── GITTINS_*_p1.0*.STOP.dat
│       ├── best_hyperparams_*.txt
│       └── tuning/
├── obj4_4_goals4_blocked1/
└── obj3_3_goals3_blocked2/
```

## Testing (Small Scale)

### Test with 10 iterations:
```bash
./dataSizeExperiment mode:generate numIterations:10 ...
./dataSizeExperiment mode:run numIterations:10 ...
```

## Common Issues

**"Configurations not found"**
→ Run `mode:generate` first

**"Results in different directories"**  
→ Make sure config family parameters match exactly

**"Want to test new data percentage"**
→ Just run `mode:run` with new percentage (configs already exist)

## File Naming Convention

- Configs: `config_{iteration}.g` and `lgp_{iteration}.lgp`
- Results: `{SOLVER}_{params}_p{data_percent}_{timestamp}.STOP.dat`
- Tuning: `tune_{hyperparams}_ELS_{params}.STOP.dat`

## Complete Workflow (Manual)

```bash
# For each family:

# 1. Generate (once)
./dataSizeExperiment -f config_base.cfg mode:generate [family_params] numIterations:100

# 2. Run GITTINS for each percentage
./dataSizeExperiment -f config_gittins.cfg mode:run [family_params] dataPercentage:0.2
./dataSizeExperiment -f config_gittins.cfg mode:run [family_params] dataPercentage:0.4
./dataSizeExperiment -f config_gittins.cfg mode:run [family_params] dataPercentage:0.6
./dataSizeExperiment -f config_gittins.cfg mode:run [family_params] dataPercentage:0.8
./dataSizeExperiment -f config_gittins.cfg mode:run [family_params] dataPercentage:1.0

# 3. Run tuning (once)
./dataSizeExperiment -f config_els_tuning.cfg mode:tune [family_params] dataPercentage:0.2
```

## Config Files

- `config_base.cfg` - Common settings
- `config_gittins.cfg` - GITTINS-specific settings
- `config_els_tuning.cfg` - ELS-specific settings

## Key Parameters

| Parameter | Typical Value | Note |
|-----------|---------------|------|
| `numIterations` | 100 | Number of configs/runs |
| `runSeed` | 0 | For reproducibility |
| `dataPercentage` | 0.2-1.0 | GNN training data |
| `solver` | GITTINS or ELS | Set in cfg file |
| `mode` | generate/run/tune | Command line |
