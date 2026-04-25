"""
Step 2: Generate plots from the saved WandB runs CSV.

Usage:
    python plot_results.py --input wandb_runs.csv --output_dir plots/

Produces one plot per model type showing average selected metric vs. data percentage,
with error bands (±1 std) across the 10 seeds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import argparse
import os

# Canonical model order matching the 7 types described in the task
MODEL_ORDER = [
    "Waypoints Feasibility",
    "Waypoints Quantile Regression Feasible",
    "Waypoints Quantile Regression Infeasible",
    "RRT Quantile Regression Feasible",
    "LGP Feasibility",
    "LGP Quantile Regression Feasible",
    "LGP Quantile Regression Infeasible",
]

# Nice colors per model
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3",
]


def _pretty_metric_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def load_and_validate(csv_path: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"model", "dataset_percent", "seed", metric}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing columns: {missing_cols}")

    # Drop runs with missing metric or dataset_percent
    n_before = len(df)
    df = df.dropna(subset=[metric, "dataset_percent"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[WARN] Dropped {n_dropped} rows with missing {metric} or dataset_percent.")

    # Only keep finished runs if state column exists
    if "state" in df.columns:
        n_before = len(df)
        df = df[df["state"] == "finished"]
        print(f"Kept {len(df)} finished runs (dropped {n_before - len(df)} non-finished).")

    df["dataset_percent"] = pd.to_numeric(df["dataset_percent"], errors="coerce")
    df = df.dropna(subset=["dataset_percent"])

    return df


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute mean and std of metric per (model, dataset_percent)."""
    agg = (
        df.groupby(["model", "dataset_percent"])[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    return agg


def plot_all_models(agg: pd.DataFrame, output_dir: str, metric: str):
    """One subplot per model, all on a single figure."""
    models_in_data = agg["model"].unique().tolist()
    # Use canonical order, fall back to whatever is in the data
    models = [m for m in MODEL_ORDER if m in models_in_data]
    models += [m for m in models_in_data if m not in models]

    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, model, color in zip(axes, models, PALETTE):
        sub = agg[agg["model"] == model].sort_values("dataset_percent")
        x = sub["dataset_percent"] * 100  # convert to percentage
        y = sub["mean"]
        err = sub["std"].fillna(0)

        ax.plot(x, y, marker="o", linewidth=2, color=color, label=model)
        ax.fill_between(x, y - err, y + err, alpha=0.2, color=color)

        # Annotate point counts
        for xi, yi, ci in zip(x, y, sub["count"]):
            ax.annotate(f"n={ci}", xy=(xi, yi), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=7, color="gray")

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training data (%)", fontsize=10)
        ax.set_ylabel(_pretty_metric_name(metric), fontsize=10)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)

    # Hide unused axes
    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.suptitle(f"{_pretty_metric_name(metric)} vs. Training Data Size\n(mean ± 1 std across 10 seeds)",
                 fontsize=14, fontweight="bold", y=1.01)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"all_models_{metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined plot: {out_path}")
    plt.close(fig)


def plot_individual(agg: pd.DataFrame, output_dir: str, metric: str):
    """One separate PNG per model."""
    models_in_data = agg["model"].unique().tolist()
    models = [m for m in MODEL_ORDER if m in models_in_data]
    models += [m for m in models_in_data if m not in models]

    os.makedirs(output_dir, exist_ok=True)

    for model, color in zip(models, PALETTE):
        sub = agg[agg["model"] == model].sort_values("dataset_percent")
        x = sub["dataset_percent"] * 100
        y = sub["mean"]
        err = sub["std"].fillna(0)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, marker="o", linewidth=2, color=color)
        ax.fill_between(x, y - err, y + err, alpha=0.2, color=color,
                        label="±1 std (across seeds)")

        for xi, yi, ci in zip(x, y, sub["count"]):
            ax.annotate(f"n={ci}", xy=(xi, yi), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=8, color="gray")

        ax.set_title(f"{model}\n{_pretty_metric_name(metric)} vs. Training Data Size",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Training data (%)", fontsize=11)
        ax.set_ylabel(_pretty_metric_name(metric), fontsize=11)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)

        fname = model.lower().replace(" ", "_") + f"_{metric}.png"
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)


def print_summary(agg: pd.DataFrame):
    print("\n=== Aggregated Results ===")
    for model in agg["model"].unique():
        sub = agg[agg["model"] == model].sort_values("dataset_percent")
        print(f"\n{model}")
        print(sub[["dataset_percent", "mean", "std", "count"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Plot metric vs. data size from WandB CSV")
    parser.add_argument("--input", default="wandb_runs.csv", help="Input CSV file (from fetch_wandb_runs.py)")
    parser.add_argument("--output_dir", default="plots", help="Directory to save plots")
    parser.add_argument(
        "--metric",
        default="test_loss",
        help="Metric column to plot (e.g., test_loss, best_val_loss)",
    )
    args = parser.parse_args()

    df = load_and_validate(args.input, args.metric)
    agg = aggregate(df, args.metric)

    print_summary(agg)
    plot_all_models(agg, args.output_dir, args.metric)
    plot_individual(agg, args.output_dir, args.metric)

    print(f"\nDone. All plots saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()