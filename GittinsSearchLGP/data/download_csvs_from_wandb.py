#!/usr/bin/env python3
"""
Download aggregated CSV files and configuration JSON from a W&B dataset artifact into
`GittinsSearchLGP/data/<folder_name>`.

Usage:
    python download_csvs_from_wandb.py <folder_name> [--project PROJECT] [--artifact-name NAME] [--version VERSION]

Example:
    python download_csvs_from_wandb.py randomBlocks_2blocks_2goals --project randomBlocks

Default artifact name (if not provided): <folder_name>-aggregated-csvs
Default version: latest

Files that will be copied (if present in the artifact):
  - aggregated_waypoints.csv
  - aggregated_rrt_by_action.csv
  - aggregated_lgp_by_plan.csv
  - aggregated_configurations.json

The script will also copy any other CSV files that are part of the artifact.
"""

import argparse
import os
import sys
import shutil
import glob
import wandb

REQUIRED_FILES = [
    "aggregated_waypoints.csv",
    "aggregated_rrt_by_action.csv",
    "aggregated_lgp_by_plan.csv",
]
CONFIG_FILE = "aggregated_configurations.json"


def resolve_dest_folder(folder_arg: str) -> str:
    # Resolve as a subfolder under this script's data directory
    base = os.path.dirname(__file__)  # points to GittinsSearchLGP/data
    dest = os.path.join(base, folder_arg)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download aggregated CSVs + config from a W&B dataset artifact")
    parser.add_argument("folder", type=str, help="Target folder name (e.g. randomBlocks_2blocks_2goals)")
    parser.add_argument("--project", type=str, default="randomBlocks", help="W&B project name")
    parser.add_argument("--artifact-name", type=str, default=None, help="Artifact name (default: <folder>-aggregated-csvs)")
    parser.add_argument("--version", type=str, default="latest", help="Artifact version (default: latest)")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (user or team)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without copying files")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Do not overwrite existing files (default: overwrite)")
    parser.set_defaults(overwrite=True)

    args = parser.parse_args()

    folder_name = args.folder
    project = args.project
    artifact_name = args.artifact_name or f"{folder_name}-aggregated-csvs"
    version = args.version
    dest_folder = resolve_dest_folder(folder_name)

    print(f"Destination folder: {dest_folder}")
    print(f"Artifact: {artifact_name}:{version} (project={project})")

    if args.dry_run:
        print("Dry run: connecting to W&B to inspect artifact contents")

    # Initialize a lightweight W&B run for artifact access
    try:
        wandb.init(project=project, entity=args.entity, job_type="dataset-download")
    except Exception as e:
        print(f"Warning: wandb.init failed: {e}. Proceeding; some API calls may still work if auth is present.")

    try:
        artifact = wandb.use_artifact(f"{artifact_name}:{version}", type="dataset")
    except Exception as e:
        print(f"Error: could not find or access artifact '{artifact_name}:{version}': {e}")
        wandb.finish()
        sys.exit(2)

    try:
        artifact_dir = artifact.download()
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        wandb.finish()
        sys.exit(3)

    print(f"Artifact downloaded to: {artifact_dir}")

    # Gather candidate files to copy
    files_to_copy = []

    # Required CSVs
    for fname in REQUIRED_FILES:
        full = os.path.join(artifact_dir, fname)
        if os.path.isfile(full):
            files_to_copy.append((full, fname))
        else:
            print(f"Warning: required file not found in artifact: {fname}")

    # Config file (copy if present)
    cfg_full = os.path.join(artifact_dir, CONFIG_FILE)
    if os.path.isfile(cfg_full):
        files_to_copy.append((cfg_full, CONFIG_FILE))
    else:
        print("Note: configuration file not found in artifact; continuing without it")

    # Also copy any other CSV files present in the artifact directory
    other_csvs = glob.glob(os.path.join(artifact_dir, "*.csv"))
    for csv_path in other_csvs:
        base = os.path.basename(csv_path)
        if base not in [f for (_, f) in files_to_copy]:
            files_to_copy.append((csv_path, base))

    if not files_to_copy:
        print("No files detected to copy from the artifact. Exiting.")
        wandb.finish()
        sys.exit(4)

    if args.dry_run:
        print("Files that would be copied:")
        for src, name in files_to_copy:
            print("  -", name, "<-", src)
        wandb.finish()
        return

    os.makedirs(dest_folder, exist_ok=True)

    # Copy files
    for src, name in files_to_copy:
        dest_path = os.path.join(dest_folder, name)
        if os.path.exists(dest_path) and not args.overwrite:
            print(f"Skipping existing file (no-overwrite): {dest_path}")
            continue
        try:
            shutil.copy2(src, dest_path)
            print(f"Copied: {name} -> {dest_path}")
        except Exception as e:
            print(f"Error copying {src} to {dest_path}: {e}")

    print(f"Download complete. Files placed in: {dest_folder}")
    wandb.finish()


if __name__ == "__main__":
    main()
