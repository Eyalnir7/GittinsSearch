"""
Generalization plot: Kendall's tau by problem family, node type, and training set.
Three panels (one per problem family), grouped bars per node type,
bars per training configuration + approximation ceiling.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "WAYPOINTS": "#5B9BD5",   # medium blue
    "LGP":       "#ED7D31",   # orange
    "RRT":       "#4CAF6E",   # green
}

NODE_LABELS = {"WAYPOINTS": "Waypoints", "LGP": "Path Opt.", "RRT": "RRT"}
NODE_ORDER  = ["WAYPOINTS", "LGP", "RRT"]

# Training-set bar configurations: (models_dir substring, label, alpha, hatch)
TRAIN_SETS = [
    ("rb2blocks_scripted$",       "RB(2,2,2) Train Set",            0.90, ""),
    ("rb2blocks3blocks_scripted$","RB(2,2,2)+RB(3,3,2) Train Set",  0.60, ""),
]
APPROX_STYLE = dict(alpha=0.30, hatch="///", edgecolor="grey", linewidth=0.6)

# Problem families
PANELS = [
    ("results_rb2b.csv", "RB(2,2,2)"),
    ("results_rb3b.csv", "RB(3,3,2)"),
    ("results_rb4b.csv", "RB(4,4,1)"),
]


def get_tau(df, node_type, models_dir_substr):
    """Return Kendall's tau for the given node type and training set."""
    task = "QUANTILE_REGRESSION_FEAS" if node_type == "RRT" else "TRIPLET"
    rows = df[
        (df["task"] == task) &
        (df["node_type"] == node_type) &
        (df["models_dir"].str.contains(models_dir_substr, na=False))
    ]
    if rows.empty:
        return np.nan
    return rows["kendall_tau"].values[0]


def get_approx(df, node_type):
    rows = df[(df["task"] == "APPROXIMATION") & (df["node_type"] == node_type)]
    if rows.empty:
        return np.nan
    return rows["kendall_tau"].values[0]


# ── load all CSVs ─────────────────────────────────────────────────────────────
panel_data = [(pd.read_csv(f), label) for f, label in PANELS]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=False)
fig.subplots_adjust(wspace=0.32)

n_groups   = len(NODE_ORDER)
n_bars     = len(TRAIN_SETS) + 1          # +1 for approximation
bar_w      = 0.18
group_span = n_bars * bar_w + 0.08        # gap between node-type groups
x_centers  = np.arange(n_groups) * group_span

for ax, (df, panel_label) in zip(axes, panel_data):

    for g, nt in enumerate(NODE_ORDER):
        color    = COLORS[nt]
        xc       = x_centers[g]
        offsets  = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w

        # Approximation bar (first slot, hatched)
        av = get_approx(df, nt)
        ax.bar(xc + offsets[0], av, bar_w,
               color=color,
               **APPROX_STYLE)
        if not np.isnan(av):
            ax.text(xc + offsets[0], av + 0.015, f"{av:.3f}",
                    ha="center", va="bottom", fontsize=7,
                    color="#555555", fontstyle="italic")

        # Training-set bars
        for b, (substr, _, alpha, hatch) in enumerate(TRAIN_SETS):
            pv = get_tau(df, nt, substr)
            slot = offsets[b + 1]
            ax.bar(xc + slot, pv, bar_w,
                   color=color, alpha=alpha, hatch=hatch,
                   edgecolor="white", linewidth=0.6)
            if not np.isnan(pv):
                ax.text(xc + slot, pv + 0.015, f"{pv:.3f}",
                        ha="center", va="bottom", fontsize=7,
                        color="#555555", fontstyle="italic")

    ax.set_title(panel_label, fontsize=12, pad=6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([NODE_LABELS[n] for n in NODE_ORDER], fontsize=10)
    ax.set_ylabel("Kendall's τ", fontsize=10)
    ax.set_ylim(-0.15, 1.10)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

# ── shared legend ─────────────────────────────────────────────────────────────
node_handles = [
    mpatches.Patch(facecolor=COLORS[n], label=NODE_LABELS[n])
    for n in NODE_ORDER
]
bar_handles = [
    mpatches.Patch(facecolor="grey", alpha=0.30, hatch="///", edgecolor="grey",
                   label="Approximation (empirical quantiles)"),
    mpatches.Patch(facecolor="grey", alpha=0.90, edgecolor="grey",
                   label="RB(2,2,2) Train Set"),
    mpatches.Patch(facecolor="grey", alpha=0.60, edgecolor="grey",
                   label="RB(2,2,2)+RB(3,3,2) Train Set"),
]
fig.legend(
    handles=node_handles + bar_handles,
    fontsize=8.5, ncol=3,
    loc="lower center", bbox_to_anchor=(0.5, -0.14),
    framealpha=0.9, edgecolor="#cccccc"
)

fig.suptitle(
    "Gittins Index Ranking Quality by Problem Family and Training Set\n"
    "(Kendall's τ; hatched = approximation ceiling)",
    fontsize=11, y=1.01
)

plt.tight_layout()
fig.savefig("plot_generalization.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved plot_generalization.png")
