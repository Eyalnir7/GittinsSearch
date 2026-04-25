"""
Group model files in a datasize folder by the seed of their training run.

The seed is not stored in the local model_meta_{run_id}.json files (those only
contain run_id, task and node_type).  Instead it is fetched from wandb:
  - First from the artifact metadata ("seed" key, if present).
  - Falling back to the run config ("seed" key), which is always set.

Usage:
    python group_models_by_seed.py datasize_0.2
    python group_models_by_seed.py datasize_0.2 --models-root all_datasize_models
    python group_models_by_seed.py datasize_0.2 --entity my_team --project randomBlocksDataSizeExp

Output example (printed to stdout as JSON):
    {
      "42": [
        "best_model_FEASIBILITY_LGP_6r5n7qrb.pt",
        "best_model_FEASIBILITY_WAYPOINTS_92bds38m.pt",
        ...
      ],
      "7": [
        ...
      ]
    }

Optionally pass --output to write the result to a JSON file.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import wandb

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
PROJECT = "randomBlocksDataSizeExp"
ENTITY = None  # None → use the currently logged-in wandb account

# Pattern: best_model_{TASK}_{NODE_TYPE}_{run_id}.pt
# The run_id is always the last 8-character alphanumeric token before ".pt".
_RUN_ID_RE = re.compile(r"_([a-z0-9]{8})$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_run_id(pt_filename: str) -> str | None:
    """Return the wandb run-id embedded in a model filename, or None."""
    stem = Path(pt_filename).stem  # strip ".pt"
    m = _RUN_ID_RE.search(stem)
    return m.group(1) if m else None


def collect_models(folder: Path) -> dict[str, dict]:
    """
    Scan *folder* for best_model_*.pt files.

    Returns a dict keyed by run_id:
        {
          run_id: {
              "pt_file":   Path,
              "task":      str | None,   # from model_meta json if available
              "node_type": str | None,
          },
          ...
        }
    """
    models: dict[str, dict] = {}

    for pt_file in sorted(folder.glob("best_model_*.pt")):
        run_id = extract_run_id(pt_file.name)
        if run_id is None:
            print(f"  [WARN] Could not extract run_id from: {pt_file.name}")
            continue
        models[run_id] = {
            "pt_file":   pt_file,
            "meta_file": None,
            "task":      None,
            "node_type": None,
        }

    # Enrich with task / node_type from the local meta json files (optional)
    for meta_file in folder.glob("model_meta_*.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        run_id = meta.get("run_id")
        if run_id and run_id in models:
            models[run_id]["task"]      = meta.get("task")
            models[run_id]["node_type"] = meta.get("node_type")
            models[run_id]["meta_file"] = meta_file

    return models


def fetch_seeds(run_ids: list[str], project: str, entity: str | None) -> dict[str, int | None]:
    """
    Query wandb for each run_id and return a mapping {run_id: seed}.

    Seed lookup order:
      1. artifact.metadata["seed"]  (logged alongside the model artifact)
      2. run.config["seed"]         (always set by the training script)
    """
    api = wandb.Api(timeout=60)
    entity_project = f"{entity}/{project}" if entity else project

    seeds: dict[str, int | None] = {}
    remaining = set(run_ids)

    print(f"\nFetching {len(remaining)} run(s) from '{entity_project}' …")

    for run_id in sorted(remaining):
        try:
            run = api.run(f"{entity_project}/{run_id}")
        except Exception as exc:
            print(f"  [WARN] Could not fetch run {run_id}: {exc}")
            seeds[run_id] = None
            continue

        seed = None

        # 1. Try artifact metadata first (user-specified storage location)
        try:
            artifacts = list(run.logged_artifacts())
            model_artifacts = [a for a in artifacts if a.type == "model"]
            if model_artifacts:
                seed = model_artifacts[0].metadata.get("seed")
        except Exception:
            pass

        # 2. Fall back to run config (always populated by the training script)
        if seed is None:
            seed = run.config.get("seed")

        if seed is None:
            print(f"  [WARN] No seed found for run {run_id} (name={run.name})")
        else:
            seed = int(seed)

        seeds[run_id] = seed
        print(f"  run={run_id}  name={run.name}  seed={seed}")

    return seeds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    folder_name: str,
    models_root: str,
    project: str,
    entity: str | None,
    output: str | None,
) -> None:
    # Resolve the target folder
    root = Path(models_root)
    if not root.is_absolute():
        root = Path(__file__).parent / root
    folder = root / folder_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f"Scanning: {folder}")
    models = collect_models(folder)
    if not models:
        print("No best_model_*.pt files found in the folder.")
        return

    print(f"Found {len(models)} model file(s).")

    # Fetch seeds from wandb
    seeds_map = fetch_seeds(list(models.keys()), project, entity)

    # Group model info by seed
    grouped: dict[int, list[dict]] = defaultdict(list)
    no_seed: list[dict] = []

    for run_id, info in sorted(models.items()):
        seed = seeds_map.get(run_id)
        if seed is None:
            no_seed.append(info)
        else:
            grouped[seed].append(info)

    # Build result dict (seed -> sorted list of filenames) for JSON output
    result = {
        str(seed): sorted(info["pt_file"].name for info in infos)
        for seed, infos in sorted(grouped.items())
    }
    if no_seed:
        result["UNKNOWN_SEED"] = sorted(info["pt_file"].name for info in no_seed)

    # Move files (pt + meta json) into seed subdirectories, then delete originals
    print(f"\n--- Moving files into seed subdirectories ({folder_name}) ---")
    all_groups = list(grouped.items()) + ([(-1, no_seed)] if no_seed else [])
    for seed_val, infos in all_groups:
        seed_key = "UNKNOWN_SEED" if seed_val == -1 else str(seed_val)
        seed_dir = folder / f"seed_{seed_key}"
        seed_dir.mkdir(exist_ok=True)
        print(f"\n  seed={seed_key}  ({len(infos)} model(s))  →  {seed_dir.relative_to(root)}")
        for info in sorted(infos, key=lambda x: x["pt_file"].name):
            for src in (info["pt_file"], info["meta_file"]):
                if src is None or not src.exists():
                    continue
                dst = seed_dir / src.name
                shutil.copy2(src, dst)
                src.unlink()
                print(f"    moved: {src.name}")

    # Serialise
    json_str = json.dumps(result, indent=2)

    if output:
        out_path = Path(output)
        out_path.write_text(json_str)
        print(f"\nGrouping written to: {out_path}")
    else:
        print(f"\n--- JSON grouping ---\n{json_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Group model files in a datasize folder by training-run seed. "
            "Seed is fetched from wandb artifact metadata / run config."
        )
    )
    parser.add_argument(
        "folder_name",
        help="Name of the datasize folder, e.g. 'datasize_0.2'.",
    )
    parser.add_argument(
        "--models-root",
        default="all_datasize_models",
        help=(
            "Root directory containing the datasize_X folders. "
            "Can be absolute or relative to this script. "
            "(default: all_datasize_models)"
        ),
    )
    parser.add_argument(
        "--project", "-p",
        default=PROJECT,
        help=f"WandB project name (default: {PROJECT})",
    )
    parser.add_argument(
        "--entity", "-e",
        default=ENTITY,
        help="WandB entity (username or team). Defaults to the currently logged-in user.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="If given, write the JSON result to this file path.",
    )

    args = parser.parse_args()
    main(
        folder_name=args.folder_name,
        models_root=args.models_root,
        project=args.project,
        entity=args.entity,
        output=args.output,
    )
