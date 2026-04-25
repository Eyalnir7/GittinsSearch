"""
Rename existing best_model_*.pt files to include the run_id suffix,
using the model_meta_*.json files as the source of truth.

Before: best_model_{TASK}_{NODE_TYPE}.pt
After:  best_model_{TASK}_{NODE_TYPE}_{run_id}.pt

Operates on all datasize_* subdirectories under --models_root.
Pass --dry-run to preview changes without actually renaming.
"""
import argparse
import json
from pathlib import Path


def rename_in_dir(ds_dir: Path, dry_run: bool) -> int:
    renames = 0
    meta_files = list(ds_dir.glob("model_meta_*.json"))
    if not meta_files:
        print(f"  [SKIP] No model_meta_*.json found in {ds_dir}")
        return 0

    for meta_path in sorted(meta_files):
        with open(meta_path) as f:
            meta = json.load(f)

        run_id    = meta["run_id"]
        task      = meta["task"]
        node_type = meta["node_type"]

        old_name = f"best_model_{task}_{node_type}.pt"
        new_name = f"best_model_{task}_{node_type}_{run_id}.pt"
        old_path = ds_dir / old_name
        new_path = ds_dir / new_name

        if new_path.exists():
            print(f"  [OK]   Already renamed: {new_name}")
            continue

        if not old_path.exists():
            print(f"  [WARN] Source not found: {old_name}  (run_id={run_id})")
            continue

        if dry_run:
            print(f"  [DRY]  {old_name}  →  {new_name}")
        else:
            old_path.rename(new_path)
            print(f"  [DONE] {old_name}  →  {new_name}")
        renames += 1

    return renames


def main():
    parser = argparse.ArgumentParser(
        description="Add run_id suffix to downloaded model .pt files."
    )
    parser.add_argument(
        "--models_root", "-r",
        default="downloaded_models",
        help="Root folder containing datasize_* subdirectories (default: downloaded_models/)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview renames without making any changes.",
    )
    args = parser.parse_args()

    root = Path(args.models_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Error: '{root}' is not a directory.")

    # Process every datasize_* subdirectory
    ds_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("datasize_"))
    if not ds_dirs:
        # Also handle the case where the user points directly at a single datasize dir
        ds_dirs = [root]

    total = 0
    for ds_dir in ds_dirs:
        print(f"\n{ds_dir.name}/")
        total += rename_in_dir(ds_dir, args.dry_run)

    action = "would rename" if args.dry_run else "renamed"
    print(f"\nDone. {action} {total} file(s).")


if __name__ == "__main__":
    main()
