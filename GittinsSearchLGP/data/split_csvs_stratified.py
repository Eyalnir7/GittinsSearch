#!/usr/bin/env python3
"""
Stratified train/val/test split of aggregated CSV files by configuration family.

The split is done at the **file_id** level (all rows that share a file_id are kept
together in the same split) and is *stratified by family*, where family = file_id.split('__')[0].
This ensures:
  - All samples from a given file_id appear in only ONE split (train, val, or test)
  - The proportion of file_ids from each family is preserved across all three splits

Usage:
    python split_csvs_stratified.py <folder> [options]

Examples:
    # Split with default 70/15/15 ratios and save to randomBlocks_all_split/
    python split_csvs_stratified.py randomBlocks_all

    # Custom ratios (must sum to 1)
    python split_csvs_stratified.py randomBlocks_all --train 0.8 --val 0.1 --test 0.1

    # Also upload each split to W&B
    python split_csvs_stratified.py randomBlocks_all --upload --project randomBlocks

    # Dry run (print stats without saving)
    python split_csvs_stratified.py randomBlocks_all --dry-run
"""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

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
# Split logic
# ---------------------------------------------------------------------------

def get_all_file_ids(folder: str) -> pd.Series:
    """Return a Series of all unique file_ids present across all CSV files."""
    all_ids: set = set()
    for fname in CSV_FILES:
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            df = pd.read_csv(path, usecols=["file_id"])
            all_ids.update(df["file_id"].astype(str).unique().tolist())
    return pd.Series(sorted(all_ids), name="file_id")


def compute_family(file_id: str) -> str:
    """Extract the family label (prefix before the first '__')."""
    return file_id.split("__")[0] if "__" in file_id else file_id


def stratified_split(
    src_folder: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Split unique file_ids into train / val / test in a stratified manner.

    For each family, shuffle its file_ids and assign them to train/val/test
    according to the requested fractions. Concatenating the per-family results
    guarantees that each family is represented proportionally in every split.
    
    All samples from the same file_id are kept together in the same split.

    Returns a dictionary mapping each CSV file name to a tuple of (train_df, val_df, test_df).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    # First, get all unique file_ids from all CSV files
    all_file_ids = get_all_file_ids(src_folder)
    
    # Create a dataframe with file_id and family mapping
    file_id_df = pd.DataFrame({
        'file_id': all_file_ids,
        'family': all_file_ids.apply(compute_family)
    })
    
    # Split file_ids (not rows) in a stratified manner by family
    # First split: train vs (val+test)
    train_file_ids, temp_file_ids = train_test_split(
        file_id_df['file_id'],
        test_size=(val_frac + test_frac),
        stratify=file_id_df['family'],
        random_state=seed
    )
    
    # Second split: val vs test
    # Get families for temp_file_ids to stratify the second split
    temp_families = file_id_df[file_id_df['file_id'].isin(temp_file_ids)]['family']
    val_file_ids, test_file_ids = train_test_split(
        temp_file_ids,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp_families,
        random_state=seed
    )
    
    # Convert to sets for fast lookup
    train_set = set(train_file_ids)
    val_set = set(val_file_ids)
    test_set = set(test_file_ids)
    
    # Now split each CSV file according to the file_id assignments
    splits = {}
    for f in CSV_FILES:
        if not os.path.isfile(os.path.join(src_folder, f)):
            raise FileNotFoundError(f"Required file not found: {f}")
        df = pd.read_csv(os.path.join(src_folder, f))
        df["family"] = df["file_id"].apply(compute_family)
        
        # Split dataframe based on file_id membership
        train_df = df[df["file_id"].isin(train_set)].copy()
        val_df = df[df["file_id"].isin(val_set)].copy()
        test_df = df[df["file_id"].isin(test_set)].copy()
        
        splits[f] = (train_df, val_df, test_df)
    
    config_path = os.path.join(src_folder, CONFIG_FILE)
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        splits[CONFIG_FILE] = (config, config, config)
    else:
        splits[CONFIG_FILE] = (None, None, None)

    return splits


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def split_stats(name: str, ids: list[str], total: int) -> None:
    families = [compute_family(fid) for fid in ids]
    fam_counts = pd.Series(families).value_counts()
    print(f"\n  [{name}]  {len(ids)} file_ids  ({100*len(ids)/total:.1f}%)")
    for fam, cnt in fam_counts.items():
        print(f"    {fam}: {cnt}")


def family_proportion_table(all_dfs: dict) -> None:
    # Use the first available CSV to report family proportions
    first_csv = next((f for f in CSV_FILES if f in all_dfs), None)
    if first_csv is None:
        print("No data to display.")
        return
    train_df, val_df, test_df = all_dfs[first_csv]
    all_families = sorted(set(train_df["family"]) | set(val_df["family"]) | set(test_df["family"]))

    def counts(df):
        c = df["family"].value_counts()
        return {fam: c.get(fam, 0) for fam in all_families}

    tr = counts(train_df)
    va = counts(val_df)
    te = counts(test_df)

    header = f"{'Family':<50} {'Train':>6} {'Val':>6} {'Test':>6}  {'Train%':>7} {'Val%':>6} {'Test%':>6}"
    print("\n" + header)
    print("-" * len(header))
    for fam in all_families:
        total = tr[fam] + va[fam] + te[fam]
        tr_p = 100 * tr[fam] / total if total else 0
        va_p = 100 * va[fam] / total if total else 0
        te_p = 100 * te[fam] / total if total else 0
        print(f"{fam:<50} {tr[fam]:>6} {va[fam]:>6} {te[fam]:>6}  {tr_p:>6.1f}% {va_p:>5.1f}% {te_p:>5.1f}%")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_splits(out_folder: str, all_dfs: dict) -> None:
    for split_name, idx in (("train", 0), ("val", 1), ("test", 2)):
        split_dir = os.path.join(out_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for fname in CSV_FILES:
            if fname not in all_dfs:
                continue
            df = all_dfs[fname][idx].drop(columns=["family"], errors="ignore").reset_index(drop=True)
            dst_path = os.path.join(split_dir, fname)
            df.to_csv(dst_path, index=False)
            print(f"  Saved {split_name}/{fname}  ({len(df)} rows, {df['file_id'].nunique()} file_ids)")

        # Write config JSON
        if CONFIG_FILE in all_dfs and all_dfs[CONFIG_FILE][idx] is not None:
            config_dst = os.path.join(split_dir, CONFIG_FILE)
            with open(config_dst, "w") as f:
                json.dump(all_dfs[CONFIG_FILE][idx], f, indent=2)
            print(f"  Saved {split_name}/{CONFIG_FILE}")


# ---------------------------------------------------------------------------
# W&B upload
# ---------------------------------------------------------------------------

def upload_to_wandb(out_folder: str, project: str, artifact_base_name: str, entity: str | None) -> None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping upload.")
        return

    for split_name in ("train", "val", "test"):
        split_dir = os.path.join(out_folder, split_name)
        if not os.path.isdir(split_dir):
            continue

        artifact_name = f"{artifact_base_name}-{split_name}"
        print(f"\nUploading {split_name} split as artifact '{artifact_name}' to project '{project}'...")

        wandb.init(project=project, entity=entity, job_type="dataset-upload", reinit=True)
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata={"split": split_name, "source_folder": os.path.abspath(split_dir)},
        )
        for fname in CSV_FILES:
            full = os.path.join(split_dir, fname)
            if os.path.isfile(full):
                artifact.add_file(full, name=fname)
        config_file = os.path.join(split_dir, CONFIG_FILE)
        if os.path.isfile(config_file):
            artifact.add_file(config_file, name=CONFIG_FILE)

        logged = wandb.log_artifact(artifact)
        try:
            logged.wait()
        except Exception:
            pass
        print(f"  Logged: {logged.name}")
        wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/val/test CSV split by configuration family")
    parser.add_argument("folder", help="Source folder containing aggregated CSV files")
    parser.add_argument("--train", type=float, default=0.80, help="Train fraction (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.1, help="Val fraction (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output folder (default: <folder>_split)")
    parser.add_argument("--upload", action="store_true", help="Upload each split to W&B after saving")
    parser.add_argument("--project", type=str, default="randomBlocksData", help="W&B project name")
    parser.add_argument("--artifact-name", type=str, default=None, help="W&B artifact base name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--dry-run", action="store_true", help="Print split stats without saving")
    args = parser.parse_args()

    # Validate fractions
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-4:
        print(f"Error: train+val+test = {total:.4f}, must equal 1.0")
        sys.exit(1)

    src_folder = resolve_folder(args.folder)
    if not os.path.isdir(src_folder):
        print(f"Error: folder not found: {src_folder}")
        sys.exit(2)

    out_folder = args.output or (src_folder.rstrip("/") + "_split")
    artifact_base = args.artifact_name or os.path.basename(src_folder.rstrip("/"))

    print(f"Source folder : {src_folder}")
    print(f"Output folder : {out_folder}")
    print(f"Split ratios  : train={args.train:.2f}  val={args.val:.2f}  test={args.test:.2f}")
    print(f"Random seed   : {args.seed}")

    # Compute splits
    print("\nComputing stratified split...")
    all_dfs = stratified_split(
        src_folder,
        train_frac=args.train,
        val_frac=args.val,
        test_frac=args.test,
        seed=args.seed,
    )



    print("\nFamily proportions across splits:")
    family_proportion_table(all_dfs)

    if args.dry_run:
        print("\nDry run — nothing saved.")
        return

    # Save
    print(f"\nSaving split CSVs to {out_folder}/...")
    os.makedirs(out_folder, exist_ok=True)
    save_splits(out_folder, all_dfs)

    if args.upload:
        upload_to_wandb(out_folder, args.project, artifact_base, args.entity)

    print("\nDone.")


if __name__ == "__main__":
    main()
