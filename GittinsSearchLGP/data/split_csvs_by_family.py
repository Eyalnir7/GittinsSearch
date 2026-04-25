#!/usr/bin/env python3
"""
Split aggregated CSV files into per-family subfolders.

Each unique family (derived from file_id via the prefix before the first '__')
gets its own subfolder containing the four data files filtered to only that family's file_ids.

Usage:
    python split_csvs_by_family.py <folder> [options]

Examples:
    # Split randomBlocks_all into randomBlocks_all_family_split/
    python split_csvs_by_family.py randomBlocks_all

    # Custom output folder
    python split_csvs_by_family.py randomBlocks_all --output my_family_split

    # Dry run (print stats without saving)
    python split_csvs_by_family.py randomBlocks_all --dry-run
"""

import argparse
import json
import os
import sys

import pandas as pd

CSV_FILES = [
    "aggregated_lgp_by_plan.csv",
    "aggregated_rrt_by_action.csv",
    "aggregated_waypoints.csv",
]
CONFIG_FILE = "aggregated_configurations.json"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_folder(folder_arg: str) -> str:
    if os.path.isabs(folder_arg) and os.path.isdir(folder_arg):
        return folder_arg
    candidate = os.path.join(os.path.dirname(__file__), folder_arg)
    if os.path.isdir(candidate):
        return candidate
    candidate2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", folder_arg))
    if os.path.isdir(candidate2):
        return candidate2
    return folder_arg


# ---------------------------------------------------------------------------
# Family helpers
# ---------------------------------------------------------------------------

def compute_family(file_id: str) -> str:
    """Extract the family label (prefix before the first '__')."""
    return file_id.split("__")[0] if "__" in file_id else file_id


def get_families(src_folder: str) -> dict[str, set[str]]:
    """
    Return a mapping from family name -> set of file_ids belonging to that family,
    derived from all unique file_ids across all CSV files.
    """
    all_ids: set[str] = set()
    for fname in CSV_FILES:
        path = os.path.join(src_folder, fname)
        if os.path.isfile(path):
            df = pd.read_csv(path, usecols=["file_id"])
            all_ids.update(df["file_id"].astype(str).unique().tolist())

    family_map: dict[str, set[str]] = {}
    for fid in all_ids:
        fam = compute_family(fid)
        family_map.setdefault(fam, set()).add(fid)
    return family_map


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------

def split_by_family(src_folder: str, out_folder: str, dry_run: bool) -> None:
    family_map = get_families(src_folder)

    if not family_map:
        print("No file_ids found — nothing to do.")
        return

    families = sorted(family_map.keys())
    print(f"\nFound {len(families)} familie(s): {', '.join(families)}")

    # Load config once (shared across all families)
    config_path = os.path.join(src_folder, CONFIG_FILE)
    config = None
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Load each CSV once
    csv_data: dict[str, pd.DataFrame] = {}
    for fname in CSV_FILES:
        path = os.path.join(src_folder, fname)
        if os.path.isfile(path):
            csv_data[fname] = pd.read_csv(path)
        else:
            print(f"  Warning: {fname} not found in {src_folder}, skipping.")

    # Per-family output
    for fam in families:
        fids = family_map[fam]
        fam_dir = os.path.join(out_folder, fam)

        print(f"\n  [{fam}]  {len(fids)} file_id(s)")

        if dry_run:
            for fname, df in csv_data.items():
                subset = df[df["file_id"].astype(str).isin(fids)]
                print(f"    {fname}: {len(subset)} rows")
            continue

        os.makedirs(fam_dir, exist_ok=True)

        for fname, df in csv_data.items():
            subset = df[df["file_id"].astype(str).isin(fids)].reset_index(drop=True)
            dst = os.path.join(fam_dir, fname)
            subset.to_csv(dst, index=False)
            print(f"    Saved {fam}/{fname}  ({len(subset)} rows)")

        if config is not None:
            dst = os.path.join(fam_dir, CONFIG_FILE)
            with open(dst, "w") as f:
                json.dump(config, f, indent=2)
            print(f"    Saved {fam}/{CONFIG_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split aggregated CSV files into per-family subfolders"
    )
    parser.add_argument("folder", help="Source folder containing aggregated CSV files")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder (default: <folder>_family_split)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without saving anything",
    )
    args = parser.parse_args()

    src_folder = resolve_folder(args.folder)
    if not os.path.isdir(src_folder):
        print(f"Error: folder not found: {src_folder}")
        sys.exit(1)

    out_folder = args.output or (src_folder.rstrip("/") + "_family_split")

    print(f"Source folder : {src_folder}")
    print(f"Output folder : {out_folder}")

    split_by_family(src_folder, out_folder, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run — nothing saved.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
