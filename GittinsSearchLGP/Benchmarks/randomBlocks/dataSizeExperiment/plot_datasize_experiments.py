"""
Datasize Experiment Plots
=========================
Generates thesis-quality plots for the datasize experiments.
Each family (obj2, obj3, obj4) was run at 5 data fractions (0.2, 0.4, 0.6, 0.8, 1.0),
with 10 independent seeds per fraction.  Each seed file has 20 problem rows.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAMILIES = {
    "obj2 (2 objs, 2 goals, 2 blocked)": "obj2_2_goals2_blocked2",
    "obj3 (3 objs, 3 goals, 2 blocked)": "obj3_3_goals3_blocked2",
    "obj4 (4 objs, 4 goals, 1 blocked)": "obj4_4_goals4_blocked1",
}
FAMILY_KEYS = list(FAMILIES.keys())
FAMILY_SHORT = ["obj2", "obj3", "obj4"]
DATA_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Colours consistent across all plots
FAM_COLORS = ["#2E86AB", "#A23B72", "#F18F01"]
DS_COLORS  = plt.cm.viridis(np.linspace(0.15, 0.85, len(DATA_SIZES)))


# ── Data loading ─────────────────────────────────────────────────────────────
def load_all_data() -> pd.DataFrame:
    """Return a tidy DataFrame with one row per problem-instance run."""
    records = []
    for fam_label, fam_dir in FAMILIES.items():
        results_dir = os.path.join(BASE, fam_dir, "results")
        for ds in DATA_SIZES:
            ds_folder = os.path.join(results_dir, f"datasize_{ds:.1f}")
            if not os.path.isdir(ds_folder):
                continue
            for fpath in glob.glob(os.path.join(ds_folder, "*.dat")):
                # Extract seed from filename  e.g. GITTINS_75_3_p0.2_ms3_...
                m = re.search(r"_ms(\d+)_", os.path.basename(fpath))
                seed = int(m.group(1)) if m else -1
                try:
                    df = pd.read_csv(fpath)
                    df["family"] = fam_label
                    df["data_frac"] = ds
                    df["seed"] = seed
                    records.append(df)
                except Exception as e:
                    print(f"Warning: could not read {fpath}: {e}")
    if not records:
        raise RuntimeError("No .dat files found – check BASE path.")
    data = pd.concat(records, ignore_index=True)
    data["planning_time"] = data["ctot"] - data["metaCtot"]
    return data


def seed_agg(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (family, data_frac, seed) → mean over problem instances."""
    cols = ["ctot", "metaCtot", "gittinsCtot", "inferenceCtot",
            "planning_time", "steps", "success"]
    agg = (
        data.groupby(["family", "data_frac", "seed"])[cols]
        .mean()
        .reset_index()
    )
    return agg


def family_stats(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (family, data_frac) → mean / std / median / q25 / q75."""
    cols = ["ctot", "metaCtot", "gittinsCtot", "inferenceCtot",
            "planning_time", "steps", "success"]
    results = []
    for (fam, ds), grp in seed_df.groupby(["family", "data_frac"]):
        row = {"family": fam, "data_frac": ds}
        for c in cols:
            row[f"{c}_mean"]   = grp[c].mean()
            row[f"{c}_std"]    = grp[c].std(ddof=1)
            row[f"{c}_median"] = grp[c].median()
            row[f"{c}_q25"]    = grp[c].quantile(0.25)
            row[f"{c}_q75"]    = grp[c].quantile(0.75)
        results.append(row)
    return pd.DataFrame(results)


# ── Helpers ──────────────────────────────────────────────────────────────────
def ax_ds_ticks(ax):
    ax.set_xticks(DATA_SIZES)
    ax.set_xticklabels([f"{int(d*100)}%" for d in DATA_SIZES])


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – Success rate vs data fraction (all families on one axes)
# ─────────────────────────────────────────────────────────────────────────────
def plot_success_rate(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        ax.errorbar(
            fam["data_frac"],
            fam["success_mean"] * 100,
            yerr=fam["success_std"] * 100,
            marker="o", linewidth=1.8, capsize=4,
            color=FAM_COLORS[i], label=short,
        )
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs Training Data Fraction")
    ax_ds_ticks(ax)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    save(fig, "1_success_rate.pdf")


def plot_success_rate_v2(stats: pd.DataFrame):
    """Bar-chart variant of the success-rate plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DATA_SIZES))
    width = 0.22
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        ax.bar(
            x + (i - 1) * width,
            fam["success_mean"] * 100,
            width=width * 0.9,
            color=FAM_COLORS[i], alpha=0.85, label=short,
            yerr=fam["success_std"] * 100, capsize=3, error_kw={"elinewidth": 1},
        )
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs Training Data Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d*100)}%" for d in DATA_SIZES])
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    save(fig, "1b_success_rate_bar.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – Total computation time (ctot) – violin per family
# ─────────────────────────────────────────────────────────────────────────────
def plot_ctot_violin(raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        fam_data = raw[raw["family"] == fam_label]
        groups = [fam_data[fam_data["data_frac"] == ds]["ctot"].values
                  for ds in DATA_SIZES]
        parts = ax.violinplot(groups, positions=DATA_SIZES, widths=0.12,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(FAM_COLORS[i])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_title(short)
        ax.set_xlabel("Training Data Fraction")
        if i == 0:
            ax.set_ylabel("Total Solve Time (s)")
        ax_ds_ticks(ax)
    fig.suptitle("Total Solve Time (ctot) vs Training Data Fraction", y=1.01)
    fig.tight_layout()
    save(fig, "2_ctot_violin.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – Total computation time (ctot) – line + CI all families
# ─────────────────────────────────────────────────────────────────────────────
def plot_ctot_line(stats: pd.DataFrame, seed_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        ax.plot(fam["data_frac"], fam["ctot_mean"],
                marker="o", color=FAM_COLORS[i], linewidth=2)
        ax.fill_between(
            fam["data_frac"],
            fam["ctot_mean"] - fam["ctot_std"],
            fam["ctot_mean"] + fam["ctot_std"],
            color=FAM_COLORS[i], alpha=0.25, label="±1 std",
        )
        ax.set_title(short)
        ax.set_xlabel("Training Data Fraction")
        if i == 0:
            ax.set_ylabel("Mean Total Solve Time (s)")
        ax_ds_ticks(ax)
    fig.suptitle("Total Solve Time (ctot) vs Training Data Fraction", y=1.01)
    fig.tight_layout()
    save(fig, "3_ctot_line.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 – Meta reasoning time (metaCtot) line all families on one axes
# ─────────────────────────────────────────────────────────────────────────────
def plot_meta_line(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        ax.errorbar(
            fam["data_frac"], fam["metaCtot_mean"],
            yerr=fam["metaCtot_std"],
            marker="s", linewidth=1.8, capsize=4,
            color=FAM_COLORS[i], label=short,
        )
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Mean Meta-Reasoning Time (s)")
    ax.set_title("Meta-Reasoning Time (metaCtot) vs Training Data Fraction")
    ax_ds_ticks(ax)
    ax.legend()
    fig.tight_layout()
    save(fig, "4_meta_line.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – Meta time breakdown (gittins + inference) stacked bars per family
# ─────────────────────────────────────────────────────────────────────────────
def plot_meta_breakdown(stats: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        x = np.arange(len(DATA_SIZES))
        w = 0.5
        ax.bar(x, fam["gittinsCtot_mean"], width=w,
               color="#5499C7", label="Gittins", alpha=0.85)
        ax.bar(x, fam["inferenceCtot_mean"], width=w,
               bottom=fam["gittinsCtot_mean"],
               color="#E59866", label="Inference", alpha=0.85)
        ax.set_title(short)
        ax.set_xlabel("Training Data Fraction")
        if i == 0:
            ax.set_ylabel("Mean Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(d*100)}%" for d in DATA_SIZES])
        if i == 0:
            ax.legend(loc="upper right")
    fig.suptitle("Meta-Reasoning Time Breakdown (Gittins + Inference)",
                 y=1.01)
    fig.tight_layout()
    save(fig, "5_meta_breakdown.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 – Meta breakdown fraction (relative) per family
# ─────────────────────────────────────────────────────────────────────────────
def plot_meta_breakdown_relative(stats: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        total = fam["gittinsCtot_mean"] + fam["inferenceCtot_mean"]
        gittins_rel = fam["gittinsCtot_mean"] / total * 100
        inference_rel = fam["inferenceCtot_mean"] / total * 100
        x = np.arange(len(DATA_SIZES))
        ax.bar(x, gittins_rel, color="#5499C7", label="Gittins", alpha=0.85)
        ax.bar(x, inference_rel, bottom=gittins_rel,
               color="#E59866", label="Inference", alpha=0.85)
        ax.set_title(short)
        ax.set_xlabel("Training Data Fraction")
        if i == 0:
            ax.set_ylabel("Proportion of Meta Time (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(d*100)}%" for d in DATA_SIZES])
        ax.set_ylim(0, 100)
        if i == 0:
            ax.legend(loc="upper right")
    fig.suptitle("Relative Meta-Reasoning Time Breakdown", y=1.01)
    fig.tight_layout()
    save(fig, "6_meta_breakdown_relative.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 – Planning time (ctot - metaCtot) line
# ─────────────────────────────────────────────────────────────────────────────
def plot_planning_time(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        ax.errorbar(
            fam["data_frac"], fam["planning_time_mean"],
            yerr=fam["planning_time_std"],
            marker="^", linewidth=1.8, capsize=4,
            color=FAM_COLORS[i], label=short,
        )
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Mean Planning Time (s)")
    ax.set_title("Planning (Non-Meta) Time vs Training Data Fraction")
    ax_ds_ticks(ax)
    ax.legend()
    fig.tight_layout()
    save(fig, "7_planning_time.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 – ctot vs metaCtot fraction of total time
# ─────────────────────────────────────────────────────────────────────────────
def plot_meta_fraction_of_total(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        meta_frac = fam["metaCtot_mean"] / fam["ctot_mean"] * 100
        ax.plot(fam["data_frac"], meta_frac, marker="D", linewidth=1.8,
                color=FAM_COLORS[i], label=short)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Meta Time / Total Time (%)")
    ax.set_title("Meta-Reasoning Time as Fraction of Total Solve Time")
    ax_ds_ticks(ax)
    ax.legend()
    fig.tight_layout()
    save(fig, "8_meta_fraction.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9 – Steps vs data fraction
# ─────────────────────────────────────────────────────────────────────────────
def plot_steps(stats: pd.DataFrame, raw: pd.DataFrame):
    # Violin + line plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        fam_raw = raw[raw["family"] == fam_label]
        groups = [fam_raw[fam_raw["data_frac"] == ds]["steps"].values
                  for ds in DATA_SIZES]
        parts = ax.violinplot(groups, positions=DATA_SIZES, widths=0.12,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(FAM_COLORS[i])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_title(short)
        ax.set_xlabel("Training Data Fraction")
        if i == 0:
            ax.set_ylabel("Steps to Solution")
        ax_ds_ticks(ax)
    fig.suptitle("Steps to Solution vs Training Data Fraction", y=1.01)
    fig.tight_layout()
    save(fig, "9_steps_violin.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 10 – Gittins time separately vs inference time (two-panel)
# ─────────────────────────────────────────────────────────────────────────────
def plot_gittins_vs_inference(stats: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label, marker in zip(
        axes,
        ["gittinsCtot_mean", "inferenceCtot_mean"],
        ["Gittins Computation Time (s)", "Inference Time (s)"],
        ["o", "s"],
    ):
        for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
            fam = stats[stats["family"] == fam_label].sort_values("data_frac")
            std_key = metric.replace("_mean", "_std")
            ax.errorbar(
                fam["data_frac"], fam[metric], yerr=fam[std_key],
                marker=marker, linewidth=1.8, capsize=4,
                color=FAM_COLORS[i], label=short,
            )
        ax.set_xlabel("Training Data Fraction")
        ax.set_ylabel(label)
        ax_ds_ticks(ax)
        ax.legend()
    axes[0].set_title("Gittins Time vs Training Data Fraction")
    axes[1].set_title("Inference Time vs Training Data Fraction")
    fig.suptitle("Meta-Reasoning Component Times", y=1.01)
    fig.tight_layout()
    save(fig, "10_gittins_inference.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 11 – Summary grid: all key metrics, one row per family
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_grid(stats: pd.DataFrame):
    metrics = [
        ("success_mean",       "success_std",       "Success Rate",          True,  100),
        ("ctot_mean",          "ctot_std",          "Total Solve Time (s)",  False, 1),
        ("metaCtot_mean",      "metaCtot_std",      "Meta Time (s)",         False, 1),
        ("planning_time_mean", "planning_time_std", "Planning Time (s)",     False, 1),
    ]
    n_fam = len(FAMILY_KEYS)
    n_met = len(metrics)
    fig, axes = plt.subplots(n_fam, n_met, figsize=(4 * n_met, 3.5 * n_fam),
                             squeeze=False)
    for col, (mean_col, std_col, ylabel, is_pct, scale) in enumerate(metrics):
        for row, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
            ax = axes[row][col]
            fam = stats[stats["family"] == fam_label].sort_values("data_frac")
            y = fam[mean_col] * scale
            yerr = fam[std_col] * scale
            ax.errorbar(DATA_SIZES, y, yerr=yerr,
                        marker="o", color=FAM_COLORS[row],
                        linewidth=1.8, capsize=4)
            ax.fill_between(DATA_SIZES, y - yerr, y + yerr,
                            color=FAM_COLORS[row], alpha=0.2)
            ax_ds_ticks(ax)
            if is_pct:
                ax.set_ylim(0, 110)
            if col == 0:
                ax.set_ylabel(f"{short}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ylabel)
            if row == n_fam - 1:
                ax.set_xlabel("Training Data Fraction")
    fig.suptitle("Summary Metrics vs Training Data Fraction", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "11_summary_grid.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 12 – Boxplot for ctot per family (box per datasize)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ctot_boxplot(raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        fam_data = raw[raw["family"] == fam_label]
        data_by_ds = [fam_data[fam_data["data_frac"] == ds]["ctot"].values
                      for ds in DATA_SIZES]
        bp = ax.boxplot(data_by_ds, positions=range(len(DATA_SIZES)),
                        patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2},
                        flierprops={"marker": ".", "markersize": 4,
                                    "alpha": 0.5})
        for patch in bp["boxes"]:
            patch.set_facecolor(FAM_COLORS[i])
            patch.set_alpha(0.65)
        ax.set_title(short)
        ax.set_xlabel("Training Data Fraction")
        if i == 0:
            ax.set_ylabel("Total Solve Time (s)")
        ax.set_xticks(range(len(DATA_SIZES)))
        ax.set_xticklabels([f"{int(d*100)}%" for d in DATA_SIZES])
    fig.suptitle("Total Solve Time Distribution vs Training Data Fraction",
                 y=1.01)
    fig.tight_layout()
    save(fig, "12_ctot_boxplot.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 13 – Normalised time: show improvement relative to 20% baseline
# ─────────────────────────────────────────────────────────────────────────────
def plot_normalised_ctot(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        fam = stats[stats["family"] == fam_label].sort_values("data_frac")
        baseline = fam[fam["data_frac"] == 0.2]["ctot_mean"].values[0]
        norm = fam["ctot_mean"] / baseline
        ax.plot(fam["data_frac"], norm, marker="o", linewidth=1.8,
                color=FAM_COLORS[i], label=short)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="baseline (20%)")
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Normalised Total Solve Time\n(relative to 20% baseline)")
    ax.set_title("Relative Change in Solve Time vs Training Data Fraction")
    ax_ds_ticks(ax)
    ax.legend()
    fig.tight_layout()
    save(fig, "13_normalised_ctot.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 14 – All families, all datasizes – scatter of inference vs gittins time
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter_gittins_inference(raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        fam_data = raw[raw["family"] == fam_label]
        sc = ax.scatter(
            fam_data["inferenceCtot"],
            fam_data["gittinsCtot"],
            c=fam_data["data_frac"],
            cmap="viridis", alpha=0.4, s=10,
            vmin=0.2, vmax=1.0,
        )
        ax.set_xlabel("Inference Time (s)")
        ax.set_ylabel("Gittins Time (s)")
        ax.set_title(short)
    cbar = fig.colorbar(sc, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label("Data Fraction")
    cbar.set_ticks(DATA_SIZES)
    cbar.set_ticklabels([f"{int(d*100)}%" for d in DATA_SIZES])
    fig.suptitle("Inference Time vs Gittins Time (coloured by Data Fraction)",
                 y=1.01)
    fig.tight_layout()
    save(fig, "14_scatter_gittins_inference.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 15 – CDF of solve time per data fraction, one subplot per family
# ─────────────────────────────────────────────────────────────────────────────
def plot_cdf_ctot(raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, (fam_label, short) in enumerate(zip(FAMILY_KEYS, FAMILY_SHORT)):
        ax = axes[i]
        for j, ds in enumerate(DATA_SIZES):
            vals = raw[(raw["family"] == fam_label) &
                       (raw["data_frac"] == ds)]["ctot"].dropna().sort_values()
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, color=DS_COLORS[j],
                    linewidth=1.5, label=f"{int(ds*100)}%")
        ax.set_xlabel("Total Solve Time (s)")
        ax.set_ylabel("CDF")
        ax.set_title(short)
        ax.set_ylim(0, 1.05)
        if i == 2:
            ax.legend(title="Data Fraction", loc="lower right")
    fig.suptitle("CDF of Total Solve Time by Training Data Fraction", y=1.01)
    fig.tight_layout()
    save(fig, "15_cdf_ctot.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    raw = load_all_data()
    print(f"  Total rows: {len(raw)}")

    seed_df = seed_agg(raw)
    stats   = family_stats(seed_df)

    print("Generating plots...")
    plot_success_rate(stats)

    plot_success_rate_v2(stats)

    plot_ctot_violin(raw)

    plot_ctot_line(stats, seed_df)

    plot_meta_line(stats)

    plot_meta_breakdown(stats)

    plot_meta_breakdown_relative(stats)

    plot_planning_time(stats)

    plot_meta_fraction_of_total(stats)

    plot_steps(stats, raw)

    plot_gittins_vs_inference(stats)

    plot_summary_grid(stats)

    plot_ctot_boxplot(raw)

    plot_normalised_ctot(stats)

    plot_scatter_gittins_inference(raw)

    plot_cdf_ctot(raw)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
