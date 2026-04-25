"""
Download the best model artifact for each (datasize, model_type) combination
from the wandb project "randomBlocksDataSizeExp".

Model types (7 total):
  waypoints  × {FEASIBILITY, QUANTILE_REGRESSION_FEAS, QUANTILE_REGRESSION_INFEAS}
  rrt        × {QUANTILE_REGRESSION_FEAS}
  lgp        × {FEASIBILITY, QUANTILE_REGRESSION_FEAS, QUANTILE_REGRESSION_INFEAS}

For each (datasize, model_type) the run with the lowest `final/best_val_loss` is chosen
and its model artifact is downloaded.

Output layout:
  <output_dir>/
    datasize_<X>/
      best_model_<TASK>_<NODE_TYPE>.pt
      best_model_<TASK>_<NODE_TYPE>_meta.json   (model_meta JSON logged alongside .pt)
"""
import argparse
import re
import os
import json
from pathlib import Path
from collections import defaultdict
import wandb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT = "randomBlocksDataSizeExp"
# Set ENTITY to your wandb username / team name, or leave None to let the API
# infer it from your currently-logged-in account.
ENTITY = None

# The 7 valid (node_type, task) combinations
VALID_COMBOS = [
    ("WAYPOINTS", "FEASIBILITY"),
    ("WAYPOINTS", "QUANTILE_REGRESSION_FEAS"),
    ("WAYPOINTS", "QUANTILE_REGRESSION_INFEAS"),
    ("RRT",       "QUANTILE_REGRESSION_FEAS"),
    ("LGP",       "FEASIBILITY"),
    ("LGP",       "QUANTILE_REGRESSION_FEAS"),
    ("LGP",       "QUANTILE_REGRESSION_INFEAS"),
]

# Map CLI task strings → task enum names used in run/artifact names
TASK_NAME_MAP = {
    "feasibility":    "FEASIBILITY",
    "feas_quantile":  "QUANTILE_REGRESSION_FEAS",
    "infeas_quantile":"QUANTILE_REGRESSION_INFEAS",
}
NODE_NAME_MAP = {
    "waypoints": "WAYPOINTS",
    "rrt":       "RRT",
    "lgp":       "LGP",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_run_name(run_name: str):
    """
    Extract (task_name, node_type) from the run name.

    Formats:
      FEASIBILITY_WAYPOINTS_s42_<run_id>
      QUANTILE_REGRESSION_FEAS_LGP_s7_10_30_50_70_90_<run_id>
    """
    # All task names in the enum
    task_names = [
        "QUANTILE_REGRESSION_FEAS",
        "QUANTILE_REGRESSION_INFEAS",
        "FEASIBILITY",
    ]
    node_types = ["WAYPOINTS", "RRT", "LGP"]

    for task in task_names:
        if run_name.startswith(task + "_"):
            remainder = run_name[len(task) + 1:]          # WAYPOINTS_s42_...
            for node in node_types:
                if remainder.startswith(node + "_"):
                    return task, node
    return None, None


def get_best_val_loss(run) -> float:
    """Return the best validation loss stored in the run summary."""
    val = run.summary.get("final/best_val_loss")
    if val is None:
        val = run.summary.get("best_val_loss")
    if val is None:
        return float("inf")
    return float(val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(output_dir: str, project: str, entity: str | None, all_runs: bool = False):
    api = wandb.Api(timeout=60)

    entity_project = f"{entity}/{project}" if entity else project
    mode_label = "all runs" if all_runs else "best run only"
    print(f"Fetching runs from '{entity_project}' … (mode: {mode_label})")
    runs = api.runs(entity_project)
    total = 0

    # Group runs by (datasize, node_type, task)
    # all_runs_map[(datasize, node_type, task)] = list of runs
    # best_runs[(datasize, node_type, task)]    = single run with lowest val loss
    all_runs_map: dict[tuple, list] = defaultdict(list)
    best_runs: dict[tuple, object] = {}
    best_losses: dict[tuple, float] = {}

    for run in runs:
        total += 1
        task_name, node_type = parse_run_name(run.name)
        if task_name is None or node_type is None:
            print(f"  [SKIP] Could not parse name: {run.name}")
            continue

        if (node_type, task_name) not in VALID_COMBOS:
            print(f"  [SKIP] Unexpected combo ({node_type}, {task_name}) in run: {run.name}")
            continue

        # datasize lives in the run config
        datasize = run.config.get("dataset_percent")
        if datasize is None:
            print(f"  [SKIP] No dataset_percent in config for run: {run.name}")
            continue

        key = (datasize, node_type, task_name)
        all_runs_map[key].append(run)

        loss = get_best_val_loss(run)
        if key not in best_losses or loss < best_losses[key]:
            best_losses[key] = loss
            best_runs[key] = run

    print(f"\nScanned {total} runs. Found {len(best_runs)} (datasize, model_type) combinations.\n")

    if not best_runs:
        print("No valid runs found – check that the project name and entity are correct.")
        return

    # Build the list of (key, run) pairs to download
    if all_runs:
        runs_to_download = [
            (key, run)
            for key in sorted(all_runs_map)
            for run in all_runs_map[key]
        ]
        print(f"Downloading ALL runs: {len(runs_to_download)} total.\n")
    else:
        runs_to_download = [
            (key, best_runs[key])
            for key in sorted(best_runs)
        ]
        print(f"Downloading BEST run per combination: {len(runs_to_download)} total.\n")

    # Download model artifact for each selected run
    output_path = Path(output_dir)
    missing = []

    for (datasize, node_type, task_name), run in runs_to_download:
        val_loss = best_losses[(datasize, node_type, task_name)]
        is_best_marker = " [BEST]" if run is best_runs.get((datasize, node_type, task_name)) else ""
        print(f"  [{datasize}] {node_type} / {task_name}  →  run={run.name}  val_loss={val_loss:.6f}{is_best_marker}")

        # Sub-directory for this datasize
        ds_dir = output_path / f"datasize_{datasize}"
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Find the model artifact logged by this run
        artifacts = list(run.logged_artifacts())
        model_artifacts = [a for a in artifacts if a.type == "model"]
        if not model_artifacts:
            print(f"    [WARN] No model artifact found for run {run.name} – skipping.")
            missing.append((datasize, node_type, task_name))
            continue

        # There should be exactly one model artifact per run; take the first
        artifact = model_artifacts[0]

        # Download into a temp dir, then copy the .pt (and meta .json) to the destination
        temp_dir = ds_dir / "_tmp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)
        artifact.download(root=str(temp_dir))

        # Rename / move the files to meaningful names
        stem = f"best_model_{task_name}_{node_type}_{run.id}"
        moved_any = False
        for f in temp_dir.iterdir():
            if f.suffix == ".pt":
                dest = ds_dir / f"{stem}.pt"
                f.rename(dest)
                moved_any = True
                print(f"    Saved: {dest.relative_to(output_path)}")
            elif f.suffix == ".json":
                dest = ds_dir / f"{stem}_meta.json"
                f.rename(dest)
                print(f"    Saved: {dest.relative_to(output_path)}")

        # Clean up temp dir
        try:
            temp_dir.rmdir()
        except OSError:
            pass  # not empty – leave it

        if not moved_any:
            print(f"    [WARN] No .pt file found in artifact for run {run.name}.")
            missing.append((datasize, node_type, task_name))
        else:
            # Write model_meta_{run_id}.json alongside the .pt file
            meta = {
                "run_id":    run.id,
                "task":      task_name,
                "node_type": node_type,
            }
            meta_path = ds_dir / f"model_meta_{run.id}.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    Saved: {meta_path.relative_to(output_path)}")

    print("\n--- Done ---")
    if missing:
        print(f"Missing artifacts for {len(missing)} combinations:")
        for m in missing:
            print(f"  {m}")
    else:
        print("All models downloaded successfully.")

    # Write a summary JSON
    summary = {}
    for (datasize, node_type, task_name), run in best_runs.items():
        key = f"{datasize}__{node_type}__{task_name}"
        entry: dict = {
            "datasize": datasize,
            "node_type": node_type,
            "task": task_name,
            "best_run_name": run.name,
            "best_run_id": run.id,
            "best_val_loss": best_losses[(datasize, node_type, task_name)],
        }
        if all_runs:
            entry["all_run_ids"] = [r.id for r in all_runs_map[(datasize, node_type, task_name)]]
        summary[key] = entry
    summary_path = output_path / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download best models from wandb project randomBlocksDataSizeExp"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="downloaded_models",
        help="Root folder where models will be saved (default: downloaded_models/)",
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=PROJECT,
        help=f"WandB project name (default: {PROJECT})",
    )
    parser.add_argument(
        "--entity", "-e",
        type=str,
        default=ENTITY,
        help="WandB entity (username or team). Defaults to the currently logged-in user.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        default=False,
        help=(
            "Download models from ALL runs (all seeds) per (datasize, model_type) combination. "
            "By default only the single best run (lowest val loss) is downloaded. "
            "With --all-runs each datasize folder will contain ~70 models instead of 7."
        ),
    )
    args = parser.parse_args()
    main(args.output_dir, args.project, args.entity, all_runs=args.all_runs)
