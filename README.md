Clone this repo and submodules
  ```
  git clone --recurse-submodules https://github.com/Eyalnir7/GittinsSearch.git
  ```

Set the `PROJECT_ROOT` environment variable to point to this repository. Run this command from the root of the cloned repository:

```
export PROJECT_ROOT="$(pwd)"
```

You can add this to your `.bashrc` or `.zshrc` to make it persistent:
```
echo 'export PROJECT_ROOT="/path/to/lgp-pddl"' >> ~/.bashrc
source ~/.bashrc
```

This variable is used throughout the project in scripts and C++ code to reference data and model directories relative to the project root.


* Compile this repo
```
cd $HOME/git/lgp-pddl

make -C rai -j1 printUbuntuAll    # for your information: what the next step will install
make -C rai -j1 installUbuntuAll  # calls sudo apt-get install; you can always interrupt

cd rai/test/LGP/pickAndPlace
make -j $(command nproc)
./x.exe
```

* Compile cmd line tools, if you like (including `lgpPlayer` and `kinEdit`)
```
cd $HOME/git/lgp-pddl

make -C rai bin
export PATH=".:$HOME/git/lgp-pddl/rai/bin:$PATH"
```

## local lib install

      export MAKEFLAGS="-j $(command nproc --ignore 2)"
      #apt update
      #apt install wget

      wget https://github.com/MarcToussaint/rai/raw/refs/heads/marc/_make/install.sh; chmod a+x install.sh
      ./install.sh ubuntu-rai
      ./install.sh libccd
      ./install.sh fcl
      ./install.sh libann
      ./install.sh rai

## LibTorch

1. Download a LibTorch package from https://pytorch.org/get-started/locally/
2. Add the /libtorch folder as a PATH variable with the name TORCH_DIR with `export TORCH_DIR=/path/to/libtorch`

# Manual for Using Gittins Search LGP Yourself

## Step 1: Generating Data
First, compile the project from the `GittinsSearchLGP/extractData` folder using make.
Run the `run_parallel_failsafe.sh` script from `GittinsSearchLGP/extractData`. You can generate instances for the family RB(i,j,k) by specifying command line arguments, or the script will prompt you interactively.

## Step 2: Preprocessing Data

The `GittinsSearchLGP/data` folder contains the raw data and preprocessing utilities.

1. **Create aggregated files**: Use `aggregate_data.sh` to generate:
   - CSV files for each node type
   - JSON file with instance descriptions (called "configurations")

2. **Split and organize data**: Use the following scripts to create the splits described in the paper:
   - `split_csvs_by_family.py` — split by instance family
   - `split_csvs_stratified.py` — stratified splitting
   - `join_datasets.py` — combine datasets
   
   Data percentage splitting will occur later when creating the PyTorch dataset.

## Step 3: Training Models

Use `Learning/train_WANDB.py` to train your models.

**Setup requirements**:
- Following previous steps you should have directories with three subdirectories: `train/`, `val/`, `test/`.
- Each subdirectory must contain these files:
  - `aggregated_configurations.json`
  - `aggregated_lgp_by_plan.csv`
  - `aggregated_rrt_by_action.csv`
  - `aggregated_waypoints.csv`
- Assign different WANDB projects for family splits vs. data percentage splits. This will help in exporting the models from wandb using the script_model.py file.

**Export trained models**: Download and script the models from WANDB artifacts using `Learning/test_compile/script_model.py`.

## Step 4: Running Gittins Search Experiments

### Setup: Model Directory Structure

Create a folder containing the 7 required models with names following the pattern:
```
model_{task}_{node_type}
```

Where:
- `task` ∈ {`FEASIBILITY`, `QUANTILE_REGRESSION_FEAS`, `QUANTILE_REGRESSION_INFEAS`}
- `node_type` ∈ {`WAYPOINTS`, `LGP`, `RRT`}

**For datasize experiments**, organize models hierarchically:
```
scriptedModels/
├── datasize_0.2/
│   ├── seed_1/
│   ├── seed_2/
│   └── ...
├── datasize_0.5/
│   └── ...
└── ...
```

(Using fractional values in [0,1] for datasize, integer values for seeds)

### Running Experiments
First, compile the project using make.
Use these scripts from `Benchmarks/randomBlocks/`:
- `run_all_experiments.sh` — main experiment runner
- `run_all_families.sh` — family-specific experiments

**Usage**: `./run_all_experiments.sh [steps to run]`
Example: `./run_all_experiments.sh 1 3`

### Experiment Workflow (Script Steps)

1. **Run script step 1** (`./run_all_experiments.sh 1`):
   - Generates 100 random instances per family
   - Run this step twice after changing the output folder's name to create 200 total instances
   - One set will be used for hyperparameter tuning, the other for testing

2. **Run script steps 3 and 4** (`./run_all_experiments.sh 3 4`):
   - Tunes hyperparameters for both Gittins Search and ELS per family

3. **Finalize tuning**:
   - For Gittins Search: Select best hyperparameters per family and set them in the `FAMILY_HYPERPARAMS` variable in the scripts

4. **Run script step 2 and all families** (`./run_all_experiments.sh 2` and `./run_all_families.sh`):
   - Generates final experimental results

**File Naming Convention:**

- Configs: `config_{iteration}.g` and `lgp_{iteration}.lgp`
- Results: `{SOLVER}_{params}_p{data_percent}_{timestamp}.STOP.dat`
- Tuning: `tune_{hyperparams}_ELS_{params}.STOP.dat`

**Directory Structure**

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
