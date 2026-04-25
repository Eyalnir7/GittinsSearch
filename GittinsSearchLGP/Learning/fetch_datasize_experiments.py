"""
Step 1: Fetch all runs from the randomBlocksDataSizeExp WandB project and save to CSV.

Usage:
    python fetch_wandb_runs.py --entity <your_wandb_entity>

You will be prompted for your API key if not already logged in,
or you can set the WANDB_API_KEY environment variable beforehand.
"""

import wandb
import pandas as pd
import re
import argparse
import os

PROJECT = "randomBlocksDataSizeTuned"

# Run name patterns (from train_WANDB.py):
#   QUANTILE_REGRESSION_FEAS_WAYPOINTS_s{seed}_p{percentiles}_{id}
#   FEASIBILITY_WAYPOINTS_s{seed}_{id}
# General: {TASK}_{NODE}_s{seed}[_p{percentiles}]_{wandb_id}
RUN_NAME_RE = re.compile(
    r"^(?P<task>FEASIBILITY|QUANTILE_REGRESSION_FEAS|QUANTILE_REGRESSION_INFEAS)"
    r"_(?P<node>WAYPOINTS|RRT|LGP)"
    r"_s(?P<seed>\d+)"
    r"(?:_p[\d_]+)?"          # optional percentiles suffix
    r"_[a-z0-9]+$"            # wandb run id suffix
)

def parse_run_name(name: str):
    """Return (task, node, seed) parsed from run name, or None if no match."""
    m = RUN_NAME_RE.match(name)
    if m:
        return m.group("task"), m.group("node"), int(m.group("seed"))
    return None


def model_label(task: str, node: str) -> str:
    """Human-readable model label combining task and node."""
    task_map = {
        "FEASIBILITY": "Feasibility",
        "QUANTILE_REGRESSION_FEAS": "Quantile Regression Feasible",
        "QUANTILE_REGRESSION_INFEAS": "Quantile Regression Infeasible",
    }
    node_map = {
        "WAYPOINTS": "Waypoints",
        "RRT": "RRT",
        "LGP": "LGP",
    }
    return f"{node_map.get(node, node)} {task_map.get(task, task)}"


def fetch_runs(entity: str) -> pd.DataFrame:
    api = wandb.Api()
    if entity is None:
        entity = api.viewer.entity
        print(f"Using default entity: {entity}")
    print(f"Fetching runs from {entity}/{PROJECT} ...")
    runs = api.runs(f"{entity}/{PROJECT}")

    records = []
    skipped = 0
    for run in runs:
        parsed = parse_run_name(run.name)
        if parsed is None:
            print(f"  [SKIP] Could not parse run name: {run.name!r}")
            skipped += 1
            continue

        task, node, seed = parsed

        # dataset_percent is stored in config
        data_pct = run.config.get("dataset_percent", None)
        if data_pct is None:
            print(f"  [WARN] No dataset_percent in config for run: {run.name!r}")

        # Metrics as logged in train_WANDB.py:
        #   test/loss
        #   final/test_loss
        #   final/best_val_loss
        summary = run.summary
        test_loss = summary.get("final/test_loss", None)
        best_val_loss = summary.get("final/best_val_loss", None)
        train_loss = summary.get("train_loss", None)

        if test_loss is None:
            test_loss = summary.get("test/loss", None)

        records.append({
            "run_id": run.id,
            "run_name": run.name,
            "task": task,
            "node": node,
            "model": model_label(task, node),
            "seed": seed,
            "dataset_percent": data_pct,
            "test_loss": test_loss,
            "best_val_loss": best_val_loss,
            "train_loss": train_loss,
            "state": run.state,
        })

    print(f"Fetched {len(records)} runs ({skipped} skipped).")
    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch WandB runs and save to CSV")
    parser.add_argument("--entity", required=False, help="WandB entity (username or team)")
    parser.add_argument("--output", default="wandb_runs.csv", help="Output CSV file path")
    args = parser.parse_args()

    # Login — will use WANDB_API_KEY env var if set, otherwise prompts interactively
    wandb.login()
    entity = args.entity or None
    df = fetch_runs(entity)

    out_path = args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} runs to: {os.path.abspath(out_path)}")
    print("\nColumn summary:")
    print(df.dtypes)
    print("\nSample rows:")
    print(df.head(10).to_string())

    # Quick sanity check
    print("\n--- Runs per model ---")
    print(df.groupby("model")["run_id"].count().to_string())
    print("\n--- Missing test_loss ---")
    missing = df["test_loss"].isna().sum()
    print(f"{missing} / {len(df)} runs have no test_loss recorded")


if __name__ == "__main__":
    main()