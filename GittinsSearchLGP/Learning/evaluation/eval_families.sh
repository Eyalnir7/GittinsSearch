#!/bin/bash

# Get PROJECT_ROOT from environment or use default
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi

python evaluate_models.py --data_dir $PROJECT_ROOT/GittinsSearchLGP/data/randomBlocks_all_split/test_family_split/randomBlocks_2blocks_2goals_2blockedgoals --models_dir $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks_scripted $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks3blocks_scripted --output results_rb2b.csv --scripted --device cpu --eval_approx --datasize_dir $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/dataSizeScripted_fullsonly

python evaluate_models.py --data_dir $PROJECT_ROOT/GittinsSearchLGP/data/randomBlocks_all_split/test_family_split/randomBlocks_4blocks_4goals_1blockedgoals --models_dir $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks_scripted $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks3blocks_scripted --output results_rb4b.csv --scripted --device cpu --eval_approx --datasize_dir $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/dataSizeScripted_fullsonly

python evaluate_models.py --data_dir $PROJECT_ROOT/GittinsSearchLGP/data/randomBlocks_all_split/test_family_split/randomBlocks_3blocks_3goals_2blockedgoals --models_dir $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks_scripted $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/rb2blocks3blocks_scripted --output results_rb3b.csv --scripted --device cpu --eval_approx --datasize_dir $PROJECT_ROOT/GittinsSearchLGP/Learning/test_compile/dataSizeScripted_fullsonly