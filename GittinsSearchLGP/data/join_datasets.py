#!/usr/bin/env python3
"""
Join aggregated datasets from multiple folders into a single combined folder.

The script concatenates these files from each input folder:
  - aggregated_waypoints.csv
  - aggregated_rrt_by_action.csv
  - aggregated_lgp_by_plan.csv
and merges the `aggregated_configurations.json` entries.

To avoid collisions between file IDs coming from different folders, the script prefixes all `file_id` values
and all keys in the JSON with the source folder name, separated by `__`.

Usage:
    python join_datasets.py --folders folder1 folder2 folder3 --output randomBlocks_all [--base-dir PATH] [--overwrite] [--dry-run]

Default base dir is the directory of this script (i.e., GittinsSearchLGP/data).
Example:
    python join_datasets.py --folders randomBlocks_2blocks_2goals randomBlocks_3blocks_3goals_2blcokedgoals randomBlocks_4blocks_4goals_1blcokedgoals --output randomBlocks_all

"""

import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path

REQUIRED_CSV_FILES = [
    "aggregated_waypoints.csv",
    "aggregated_rrt_by_action.csv",
    "aggregated_lgp_by_plan.csv",
]
CONFIG_FILE = "aggregated_configurations.json"
PREFIX_SEP = "__"


def resolve_folder(base_dir: str, folder_name: str) -> str:
    # Try absolute, then relative to base_dir
    if os.path.isabs(folder_name) and os.path.isdir(folder_name):
        return folder_name
    candidate = os.path.join(base_dir, folder_name)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Folder not found: {folder_name} (checked: {candidate})")


def prefix_file_id_series(series: pd.Series, prefix: str) -> pd.Series:
    # Convert to str and prefix
    return prefix + PREFIX_SEP + series.astype(str)


def main():
    parser = argparse.ArgumentParser(description="Join multiple aggregated dataset folders into a single combined folder")
    parser.add_argument("--folders", nargs="+", required=True, help="List of source folder names (relative to base-dir or absolute paths)")
    parser.add_argument("--output", type=str, default="randomBlocks_all", help="Name of the output folder to create under base-dir")
    parser.add_argument("--base-dir", type=str, default=os.path.dirname(__file__), help="Base directory containing the dataset folders (default: script directory)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output folder if present")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_folder = os.path.join(base_dir, args.output)

    # Validate source folders
    sources = []
    for f in args.folders:
        try:
            resolved = resolve_folder(base_dir, f)
        except FileNotFoundError as e:
            print(e)
            sys.exit(2)
        sources.append((f, resolved))  # (folder_name, abs_path)

    # Check output folder
    if os.path.exists(output_folder):
        if args.overwrite:
            print(f"Overwriting existing output folder: {output_folder}")
        else:
            print(f"Error: output folder already exists: {output_folder}. Use --overwrite to replace it.")
            sys.exit(3)

    # Collect DataFrames
    combined_dfs = {fname: [] for fname in REQUIRED_CSV_FILES}
    combined_config = {}

    for folder_name, folder_path in sources:
        print(f"Processing source: {folder_name} -> {folder_path}")

        # CSVs
        for csv_name in REQUIRED_CSV_FILES:
            csv_path = os.path.join(folder_path, csv_name)
            if not os.path.isfile(csv_path):
                print(f"Warning: {csv_name} not found in {folder_path}; skipping")
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading CSV {csv_path}: {e}")
                sys.exit(4)

            # If file_id column exists, prefix it
            if 'file_id' in df.columns:
                df['file_id'] = prefix_file_id_series(df['file_id'], folder_name)
            else:
                print(f"Warning: 'file_id' column not in {csv_path}; rows will be copied without modification")

            combined_dfs[csv_name].append(df)
            print(f"  Read {len(df)} rows from {csv_name}")

        # Config JSON
        cfg_path = os.path.join(folder_path, CONFIG_FILE)
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"Error reading config {cfg_path}: {e}")
                sys.exit(5)

            # Prefix keys and add to combined_config
            for k, v in cfg.items():
                new_key = folder_name + PREFIX_SEP + str(k)
                if new_key in combined_config:
                    print(f"Warning: key collision for {new_key} in combined_config; overwriting")
                combined_config[new_key] = v
            print(f"  Read {len(cfg)} config entries from {cfg_path}")
        else:
            print(f"Warning: {CONFIG_FILE} not found in {folder_path}; skipping")

    # Summarize
    total_rows = {k: sum(len(df) for df in v_list) for k, v_list in combined_dfs.items()}
    total_configs = len(combined_config)
    print("\nSummary of collected data:")
    for csv_name, rows in total_rows.items():
        print(f"  {csv_name}: {rows} rows")
    print(f"  {CONFIG_FILE}: {total_configs} entries")

    if args.dry_run:
        print("Dry run: no files written")
        return

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Write concatenated CSVs
    for csv_name, df_list in combined_dfs.items():
        if not df_list:
            print(f"Skipping {csv_name}: no data collected")
            continue
        combined = pd.concat(df_list, ignore_index=True, sort=False)
        out_path = os.path.join(output_folder, csv_name)
        combined.to_csv(out_path, index=False)
        print(f"Wrote {len(combined)} rows to {out_path}")

    # Write combined config
    out_cfg_path = os.path.join(output_folder, CONFIG_FILE)
    with open(out_cfg_path, 'w', encoding='utf-8') as f:
        json.dump(combined_config, f, indent=2)
    print(f"Wrote {len(combined_config)} entries to {out_cfg_path}")

    print(f"Done. Combined dataset available at: {output_folder}")


if __name__ == "__main__":
    main()
