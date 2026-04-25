"""
Evaluation plots for Gittins Index ranking quality (Kendall's tau).
Generates three publication-quality figures for thesis Chapter 6:

  Fig A – Bar chart: Kendall's tau by node type and model variant
          (Approximation vs GNN-Predicted), with approximation ceiling shown.
  Fig B – Line+ribbon chart: Kendall's tau vs. training-data fraction,
          per node type (mean ± 1 std, seeds as scatter).
  Fig C – Violin + strip chart: per-seed distribution of Kendall's tau at
          full data (100 %), comparing node types.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── colour palette matching the thesis plots ─────────────────────────────────
COLORS = {
    "WAYPOINTS":      "#5B9BD5",   # medium blue
    "LGP":            "#ED7D31",   # orange
    "RRT":            "#4CAF6E",   # green
    "APPROX":         "#A9D18E",   # sage green  (approximation ceiling)
    "approx_hatch":   "///",
}

NODE_LABELS = {"WAYPOINTS": "Waypoints", "LGP": "Path Opt.", "RRT": "RRT"}
NODE_ORDER  = ["WAYPOINTS", "LGP", "RRT"]

# ── load data ─────────────────────────────────────────────────────────────────
main   = pd.read_csv("evaluation_results.csv")
detail = pd.read_csv("evaluation_results_datasize_detail.csv")

# Normalise task names for the detail frame
detail["task_clean"] = detail["task"].apply(
    lambda t: "TRIPLET" if t.startswith("TRIPLET") else t
)

# Pull out the three data slices we need
approx_rows = main[main["task"] == "APPROXIMATION"].copy()
agg_rows    = main[main["task"] == "TRIPLET_DATASIZE_AGG"].copy()

# Full-data (datasize_p=1.0) aggregate results
fulldata_rows = main[
    (main["task"] == "TRIPLET_DATASIZE_AGG") &
    (main["datasize_p"] == 1.0)
].copy()

# ═════════════════════════════════════════════════════════════════════════════
# Figure A – Bar chart: Approximation ceiling vs. GNN prediction
# ═════════════════════════════════════════════════════════════════════════════
fig_a, ax = plt.subplots(figsize=(7, 4.5))

x       = np.arange(len(NODE_ORDER))
bar_w   = 0.32
approx_vals = {
    row["node_type"]: row["kendall_tau"]
    for _, row in approx_rows.iterrows()
}
pred_vals = {}
for _, row in fulldata_rows.iterrows():
    pred_vals[row["node_type"]] = row["kendall_tau_mean"]

for i, nt in enumerate(NODE_ORDER):
    av = approx_vals.get(nt, np.nan)
    pv = pred_vals.get(nt, np.nan)
    color = COLORS[nt]

    # Approximation bar (hatched)
    ax.bar(x[i] - bar_w / 2, av, bar_w,
           color=color, alpha=0.45, edgecolor="white", linewidth=0.8,
           hatch="///", label="_approx")

    # Prediction bar (solid)
    ax.bar(x[i] + bar_w / 2, pv, bar_w,
           color=color, alpha=0.92, edgecolor="white", linewidth=0.8,
           label="_pred")

    # Exact-value annotations on each bar
    if not np.isnan(av):
        ax.text(x[i] - bar_w / 2, av + 0.012, f"{av:.3f}",
                ha="center", va="bottom", fontsize=8,
                color="#444444", fontstyle="italic")
    if not np.isnan(pv):
        ax.text(x[i] + bar_w / 2, pv + 0.012, f"{pv:.3f}",
                ha="center", va="bottom", fontsize=8,
                color="#444444", fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels([NODE_LABELS[n] for n in NODE_ORDER], fontsize=11)
ax.set_ylabel("Kendall's τ", fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_title(
    "Gittins Index Ranking Quality by Node Type\n"
    "(Approximation vs. GNN-Predicted Bandit Process)",
    fontsize=11, pad=8
)
ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    mpatches.Patch(facecolor="grey", alpha=0.45, hatch="///", edgecolor="grey",
                   label="Approximation (empirical quantiles)"),
    mpatches.Patch(facecolor="grey", alpha=0.92, edgecolor="grey",
                   label="GNN Prediction"),
] + [
    mpatches.Patch(facecolor=COLORS[n], label=NODE_LABELS[n])
    for n in NODE_ORDER
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
fig_a.savefig("plot_eval_A_approx_vs_pred.png", dpi=180, bbox_inches="tight")
plt.close(fig_a)
print("Saved plot_eval_A_approx_vs_pred.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure B – Learning curve: Kendall's τ vs. training-data fraction
# ═════════════════════════════════════════════════════════════════════════════
fracs = sorted(agg_rows["datasize_p"].unique())

fig_b, ax = plt.subplots(figsize=(7.5, 4.5))

for nt in NODE_ORDER:
    color  = COLORS[nt]
    means, stds, medians = [], [], []

    for f in fracs:
        row = agg_rows[(agg_rows["node_type"] == nt) & (agg_rows["datasize_p"] == f)]
        if row.empty:
            means.append(np.nan); stds.append(np.nan); medians.append(np.nan)
        else:
            means.append(row["kendall_tau_mean"].values[0])
            stds.append(row["kendall_tau_std"].values[0])
            medians.append(row["kendall_tau_median"].values[0])

    means   = np.array(means)
    stds    = np.array(stds)
    fracs_x = np.array(fracs) * 100   # → percent

    # ribbon ± 1 std
    ax.fill_between(fracs_x, means - stds, means + stds,
                    color=color, alpha=0.18)
    # mean line
    ax.plot(fracs_x, means, "o-", color=color, linewidth=2,
            markersize=6, label=NODE_LABELS[nt], zorder=4)
    # median dashed
    ax.plot(fracs_x, medians, "--", color=color, linewidth=1.2,
            alpha=0.7, zorder=3)

    # individual seed scatter (light)
    for f in fracs:
        task_col = "TRIPLET" if nt != "RRT" else "QUANTILE_REGRESSION_FEAS"
        sub = detail[
            (detail["task_clean"] == task_col) &
            (detail["node_type"] == nt) &
            (detail["datasize_p"] == f)
        ]
        jitter = (np.random.default_rng(42).random(len(sub)) - 0.5) * 3
        ax.scatter(f * 100 + jitter, sub["kendall_tau"],
                   color=color, s=14, alpha=0.35, zorder=2)

ax.set_xlabel("Training Data Fraction (%)", fontsize=11)
ax.set_ylabel("Kendall's τ", fontsize=11)
ax.set_xticks(np.array(fracs) * 100)
ax.set_xticklabels([f"{int(f*100)}%" for f in fracs], fontsize=10)
ax.set_ylim(0.28, 0.88)
ax.set_title(
    "Gittins Index Ranking Quality vs. Training Data Size\n"
    "(mean ± 1 std, dashed = median)",
    fontsize=11, pad=8
)
ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

# legend: node types + style guide
style_handles = [
    Line2D([0], [0], color="grey", linewidth=2, marker="o", markersize=6,
           label="Mean"),
    Line2D([0], [0], color="grey", linewidth=1.2, linestyle="--", alpha=0.7,
           label="Median"),
    mpatches.Patch(facecolor="grey", alpha=0.18, label="± 1 std"),
]
node_handles = [
    Line2D([0], [0], color=COLORS[n], linewidth=2.5, label=NODE_LABELS[n])
    for n in NODE_ORDER
]
ax.legend(handles=node_handles + style_handles, fontsize=8.5, ncol=2,
          loc="lower right", framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
fig_b.savefig("plot_eval_B_learning_curve.png", dpi=180, bbox_inches="tight")
plt.close(fig_b)
print("Saved plot_eval_B_learning_curve.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure C – Violin + strip: seed distribution at 100 % data
# ═════════════════════════════════════════════════════════════════════════════
full_data = {}
for nt in NODE_ORDER:
    task_col = "TRIPLET" if nt != "RRT" else "QUANTILE_REGRESSION_FEAS"
    sub = detail[
        (detail["task_clean"] == task_col) &
        (detail["node_type"] == nt) &
        (detail["datasize_p"] == 1.0)
    ]["kendall_tau"].dropna().values
    full_data[nt] = sub

fig_c, ax = plt.subplots(figsize=(6.5, 4.5))

x_pos = np.arange(len(NODE_ORDER))
rng   = np.random.default_rng(0)

for i, nt in enumerate(NODE_ORDER):
    vals  = full_data[nt]
    color = COLORS[nt]

    if len(vals) >= 3:
        parts = ax.violinplot(vals, positions=[x_pos[i]], widths=0.5,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.0)

    # box
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    ax.vlines(x_pos[i], q25, q75, color=color, linewidth=5, alpha=0.6,
              zorder=4)
    ax.scatter(x_pos[i], q50, color="white", s=35, zorder=5,
               edgecolors=color, linewidths=1.5)

    # strip
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(x_pos[i] + jitter, vals, color=color, s=28, alpha=0.7,
               zorder=6, edgecolors="white", linewidths=0.4)

    # approximation ceiling
    av = approx_vals.get(nt, np.nan)
    if not np.isnan(av):
        ax.hlines(av, x_pos[i] - 0.3, x_pos[i] + 0.3,
                  color=color, linestyle=(0, (4, 2)), linewidth=1.5,
                  alpha=0.8, zorder=7)
        ax.text(x_pos[i] + 0.32, av, f"Approx. τ={av:.3f}",
                va="center", fontsize=7.5, color=color, alpha=0.9)

ax.set_xticks(x_pos)
ax.set_xticklabels([NODE_LABELS[n] for n in NODE_ORDER], fontsize=11)
ax.set_ylabel("Kendall's τ  (100 % training data, 10 seeds)", fontsize=10)
ax.set_ylim(0.28, 1.05)
ax.set_title(
    "Distribution of Gittins Index Ranking Quality\nat Full Training Data (seed variability)",
    fontsize=11, pad=8
)
ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    mpatches.Patch(facecolor="grey", alpha=0.35, label="Violin (density)"),
    Line2D([0], [0], color="grey", linewidth=5, alpha=0.6,
           label="IQR (25–75 %)"),
    Line2D([0], [0], color="white", marker="o", markersize=7,
           markeredgecolor="grey", markeredgewidth=1.5, label="Median"),
    Line2D([0], [0], color="grey", linestyle=(0, (4, 2)), linewidth=1.5,
           alpha=0.8, label="Approx. ceiling"),
] + [
    mpatches.Patch(facecolor=COLORS[n], label=NODE_LABELS[n])
    for n in NODE_ORDER
]
ax.legend(handles=legend_handles, fontsize=8.5, ncol=2, loc="lower right",
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
fig_c.savefig("plot_eval_C_violin_full_data.png", dpi=180, bbox_inches="tight")
plt.close(fig_c)
print("Saved plot_eval_C_violin_full_data.png")

print("\nAll three evaluation plots generated successfully.")
