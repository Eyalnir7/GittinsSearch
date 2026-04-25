#!/usr/bin/env python3
"""
Upload aggregated CSV files to Weights & Biases as an artifact.

Usage:
    python upload_csvs_to_wandb.py <folder_name> [--project PROJECT] [--artifact-name NAME]

Example:
    python upload_csvs_to_wandb.py randomBlocks_2blocks_2goals --project randomBlocks --artifact-name randomBlocks_2blocks_2goals_csvs

The script looks for the files:
  - aggregated_waypoints.csv
  - aggregated_rrt_by_action.csv
  - aggregated_lgp_by_plan.csv
in the specified folder (either absolute path or a subfolder of this script's directory). The script will also upload
`aggregated_configurations.json` if it exists in the folder."""

import argparse
import os
import sys
import wandb

REQUIRED_FILES = [
    "aggregated_waypoints.csv",
    "aggregated_rrt_by_action.csv",
    "aggregated_lgp_by_plan.csv",
]


def resolve_folder(folder_arg: str) -> str:
    # If absolute or exists as provided, use it. Otherwise, resolve relative to this script's directory.
    if os.path.isabs(folder_arg) and os.path.isdir(folder_arg):
        return folder_arg
    candidate = os.path.join(os.path.dirname(__file__), folder_arg)
    if os.path.isdir(candidate):
        return candidate
    # also try up one level (common when running from different cwd)
    candidate2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", folder_arg))
    if os.path.isdir(candidate2):
        return candidate2
    return folder_arg  # return as-is; caller will validate


def main():
    parser = argparse.ArgumentParser(description="Upload aggregated CSV files to a W&B dataset artifact")
    parser.add_argument("folder", type=str, help="Folder containing aggregated CSV files (e.g. randomBlocks_2blocks_2goals)")
    parser.add_argument("--project", type=str, default="randomBlocksData", help="W&B project name")
    parser.add_argument("--artifact-name", type=str, default=None, help="Artifact name to create (default derived from folder)")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (user or team) to log artifact to")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without calling W&B")

    # Note: the script will always attempt to upload `aggregated_configurations.json` if it exists in the folder

    args = parser.parse_args()
    folder_path = resolve_folder(args.folder)

    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(2)

    # Check required files
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(folder_path, f))]
    if missing:
        print("Error: the following required files are missing in the folder:")
        for m in missing:
            print("  -", m)
        sys.exit(3)

    # Optionally include config file
    config_path = os.path.join(folder_path, "aggregated_configurations.json")
    has_config = os.path.isfile(config_path)

    artifact_name = args.artifact_name or f"{os.path.basename(folder_path)}-aggregated-csvs"

    print(f"Folder resolved to: {folder_path}")
    print(f"Uploading files to W&B project: {args.project}, artifact name: {artifact_name}")
    if args.dry_run:
        print("Dry run: the following files would be uploaded:")
        for f in REQUIRED_FILES:
            print("  ", os.path.join(folder_path, f))
        if args.include_config and has_config:
            print("  ", config_path)
        return

    # Init W&B run
    wandb.init(project=args.project, entity=args.entity, job_type="dataset-upload")

    artifact = wandb.Artifact(name=artifact_name, type="dataset", metadata={
        "source_folder": os.path.abspath(folder_path),
    })

    # Add CSV files
    for f in REQUIRED_FILES:
        full = os.path.join(folder_path, f)
        artifact.add_file(full, name=f)
        print(f"Added: {full}")

    # Always attempt to add aggregated_configurations.json if available
    if has_config:
        artifact.add_file(config_path, name="aggregated_configurations.json")
        print(f"Added config: {config_path}")
    else:
        print("Note: aggregated_configurations.json was not found in folder; continuing without it")

    # Log artifact
    logged_artifact = wandb.log_artifact(artifact)
    try:
        logged_artifact.wait()
    except Exception:
        # Some older wandb clients might not support wait(); ignore quietly
        pass

    print(f"Successfully logged artifact: {logged_artifact.name} (version: {getattr(logged_artifact, 'version', 'unknown')})")

    wandb.finish()


if __name__ == "__main__":
    main()
