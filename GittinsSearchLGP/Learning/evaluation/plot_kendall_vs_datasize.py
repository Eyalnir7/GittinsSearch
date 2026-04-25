"""
plot_kendall_vs_datasize.py

Plots Kendall-tau vs. datasize for each node type, reading from
eval_results_testdata_{p}.csv files.

  - WAYPOINTS:  row where task == "TRIPLET"
  - LGP:        row where task == "TRIPLET"
  - RRT:        row where task == "QUANTILE_REGRESSION_FEAS"

Usage
-----
python plot_kendall_vs_datasize.py [--results_dir .] [--output kendall_vs_datasize.png]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Which row to pick per node_type
# ---------------------------------------------------------------------------
NODE_TYPE_TASK = {
    "WAYPOINTS": "TRIPLET",
    "LGP":       "TRIPLET",
    "RRT":       "QUANTILE_REGRESSION_FEAS",
}

NODE_TYPE_LABELS = {
    "WAYPOINTS": "Waypoints (triplet)",
    "LGP":       "LGP (triplet)",
    "RRT":       "RRT (quantile feas)",
}

COLORS = {
    "WAYPOINTS": "#1f77b4",
    "LGP":       "#2ca02c",
    "RRT":       "#d62728",
}


def parse_datasize(stem: str) -> float:
    """Extract the numeric datasize from 'eval_results_testdata_{p}'."""
    m = re.search(r"eval_results_testdata_(.+)$", stem)
    if m is None:
        raise ValueError(f"Cannot parse datasize from filename stem: {stem!r}")
    return float(m.group(1))


def load_results(results_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """
    Returns {node_type: [(datasize, kendall_tau), ...]} sorted by datasize.
    """
    csv_files = sorted(results_dir.glob("eval_results_testdata_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No 'eval_results_testdata_*.csv' files found in {results_dir}"
        )

    data: dict[str, list] = {nt: [] for nt in NODE_TYPE_TASK}

    for csv_path in csv_files:
        try:
            datasize = parse_datasize(csv_path.stem)
        except ValueError as e:
            print(f"  WARNING: {e}. Skipping {csv_path.name}.")
            continue

        df = pd.read_csv(str(csv_path))

        for node_type, task in NODE_TYPE_TASK.items():
            mask = (df["node_type"] == node_type) & (df["task"] == task)
            rows = df[mask]
            if rows.empty:
                print(f"  WARNING: No '{task}' row for {node_type} in {csv_path.name}.")
                continue
            # In case of duplicates, take the first
            tau = float(rows.iloc[0]["kendall_tau"])
            data[node_type].append((datasize, tau))

    # Sort by datasize
    for nt in data:
        data[nt].sort(key=lambda x: x[0])

    return data


def plot(data: dict[str, list], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Kendall-τ vs. Training Data Size", fontsize=14, fontweight="bold")

    node_types = ["WAYPOINTS", "LGP", "RRT"]

    for ax, node_type in zip(axes, node_types):
        points = data[node_type]
        if not points:
            ax.set_title(NODE_TYPE_LABELS[node_type])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        xs, ys = zip(*points)
        color = COLORS[node_type]

        ax.plot(xs, ys, marker="o", linewidth=2, markersize=7,
                color=color, label=NODE_TYPE_LABELS[node_type])
        ax.scatter(xs, ys, color=color, zorder=5)

        # Annotate each point with its value
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.3f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        ax.set_title(NODE_TYPE_LABELS[node_type], fontsize=11)
        ax.set_xlabel("Data size (fraction)", fontsize=10)
        ax.set_ylabel("Kendall-τ", fontsize=10)
        ax.set_xticks(xs)
        ax.set_ylim(bottom=max(0, min(ys) - 0.1), top=min(1, max(ys) + 0.12))
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(output), dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Kendall-tau vs. datasize for each node type."
    )
    parser.add_argument(
        "--results_dir",
        default=".",
        help="Directory containing eval_results_testdata_*.csv files (default: current dir).",
    )
    parser.add_argument(
        "--output",
        default="kendall_vs_datasize.png",
        help="Output image path (default: kendall_vs_datasize.png).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output      = Path(args.output)

    print(f"Reading results from: {results_dir}")
    data = load_results(results_dir)

    for nt, points in data.items():
        print(f"  {nt}: {[(round(x,2), round(y,4)) for x,y in points]}")

    plot(data, output)


if __name__ == "__main__":
    main()
