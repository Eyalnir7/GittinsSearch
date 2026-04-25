#!/usr/bin/env bash

set -e  # stop on first error

folders=(
  randomBlocks_2blocks_2goals_2blockedgoals
  randomBlocks_3blocks_3goals_2blockedgoals
  randomBlocks_4blocks_4goals_1blockedgoals
)

for folder in "${folders[@]}"; do
  echo "Processing $folder"

  python aggregate_data.py "$folder"
  python aggregate_lgp_further.py "$folder"
  python aggregate_rrt_further.py "$folder"
  python aggregate_conf_files.py "$folder"

  echo "Done with $folder"
  echo "-------------------------"
done
