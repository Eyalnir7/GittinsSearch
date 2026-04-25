"""
Comparison Experiment Plots
============================
Generates thesis-quality plots comparing ELS vs. Gittins solvers
trained on different dataset sizes, across three problem families.

Families:
  obj2_2_goals2_blocked2  → RB(2,2,2)
  obj3_3_goals3_blocked2  → RB(3,3,2)
  obj4_4_goals4_blocked1  → RB(4,4,1)

Solvers compared (from per-family result files in results/):
  ELS              – blind search baseline (no meta-reasoning)
  rb2bTrain        – Gittins trained on RB(2,2,2) train set
  rb2b3bTrain      – Gittins trained on RB(2,2,2)+RB(3,3,2) train set
  TunedGittins     – Gittins trained on Full train set
"""

import os
import glob
import re
import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "plots_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Family definitions ───────────────────────────────────────────────────────
FAMILIES = {
    "RB(2,2,2)": "obj2_2_goals2_blocked2",
    "RB(3,3,2)": "obj3_3_goals3_blocked2",
    "RB(4,4,1)": "obj4_4_goals4_blocked1",
}
FAMILY_LABELS = list(FAMILIES.keys())   # x-axis group names

# ── Solver definitions ───────────────────────────────────────────────────────
# Each entry: (solver_key, glob_pattern, display_label)
SOLVER_DEFS = [
    ("ELS",          "*ELS*.dat",                "ELS"),
    ("rb2bTrain",    "rb2bTrain_*.dat",           "RB(2,2,2) Train Set"),
    ("rb2b3bTrain",  "rb2b3bTrain_*.dat",         "RB(2,2,2)+RB(3,3,2) Train Set"),
    ("TunedGittins", "TunedGittinsFullRun_*.dat", "Full Train Set"),
]

SOLVER_KEYS   = [s[0] for s in SOLVER_DEFS]
SOLVER_LABELS = {s[0]: s[2] for s in SOLVER_DEFS}

# ── Colour palette ── (matches plot_results.py style) ───────────────────────
# Per-solver base colours (used for the planning-time portion of each bar)
SOLVER_COLORS = {
    "ELS":          "#d62728",   # red
    "rb2bTrain":    "#6baed6",   # light blue
    "rb2b3bTrain":  "#2171b5",   # mid blue
    "TunedGittins": "#084594",   # dark blue
}

# Fixed colours for the meta-reasoning decomposition stripes
# (applied consistently across all Gittins bars)
COLOR_INFERENCE = "#F18F01"   # amber  – inference time
COLOR_GITTINS   = "#27AE60"   # green  – Gittins index computation time


# ── Data loading ─────────────────────────────────────────────────────────────
def find_solver_file(results_dir: str, pattern: str) -> str | None:
    """Return the *latest* file (by name sort) matching pattern, or None."""
    matches = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not matches:
        return None
    # For ELS prefer the standalone `ELS_...p1.0...` run over tuning files;
    # if none such exists fall back to any match.
    if "ELS" in pattern:
        proper = [m for m in matches if os.path.basename(m).startswith("ELS_")]
        if proper:
            return sorted(proper)[-1]
    return matches[-1]   # latest by filename (timestamp suffix)


def load_file(path: str) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        df["planning_time"] = df["ctot"]
        df["ctot"] = df["planning_time"] + df["metaCtot"]
        return df
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


def load_all() -> dict[str, dict[str, pd.DataFrame | None]]:
    """Return data[family_label][solver_key] = DataFrame (or None)."""
    data = {}
    for fam_label, fam_dir in FAMILIES.items():
        results_dir = os.path.join(BASE, fam_dir, "results")
        data[fam_label] = {}
        for key, pattern, _ in SOLVER_DEFS:
            fpath = find_solver_file(results_dir, pattern)
            df    = load_file(fpath)
            data[fam_label][key] = df
            status = os.path.basename(fpath) if fpath else "NOT FOUND"
            print(f"  {fam_label:12s}  {key:16s}  {status}")
    return data


DATA_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Colours for data-fraction bars: light → dark blue gradient
DATASIZE_COLORS = {
    0.2: "#c6dbef",
    0.4: "#9ecae1",
    0.6: "#6baed6",
    0.8: "#2171b5",
    1.0: "#084594",
}
DATASIZE_LABELS = {ds: f"{int(ds*100)}% data" for ds in DATA_SIZES}


def load_all_datasizes() -> dict[str, dict[float, pd.DataFrame | None]]:
    """Return data[family_label][data_frac] = concatenated DataFrame over all seeds."""
    data = {}
    for fam_label, fam_dir in FAMILIES.items():
        results_dir = os.path.join(BASE, fam_dir, "results")
        data[fam_label] = {}
        for ds in DATA_SIZES:
            ds_folder = os.path.join(results_dir, f"datasize_{ds:.1f}")
            frames = []
            for fpath in sorted(glob.glob(os.path.join(ds_folder, "*.dat"))):
                df = load_file(fpath)
                if df is not None:
                    frames.append(df)
            combined = pd.concat(frames, ignore_index=True) if frames else None
            data[fam_label][ds] = combined
            n = len(frames) if frames else 0
            print(f"  {fam_label:12s}  ds={ds:.1f}  {n} seed files")
    return data


def filter_to_successful_instances(data: dict, ds_data: dict):
    """
    Filter all dataframes to only rows where success == 1 in ALL datasets for each family.
    Modifies data and ds_data in place.
    """
    for fam_label in FAMILY_LABELS:
        # Collect all successful indices across all solvers and datafractions
        successful_indices = None
        
        # Collect success indices from main solvers
        if fam_label in data:
            for solver_key in SOLVER_KEYS:
                df = data[fam_label].get(solver_key)
                if df is not None and "success" in df.columns:
                    successful_mask = df["success"] == 1
                    if successful_indices is None:
                        successful_indices = successful_mask.index[successful_mask].to_list()
                    else:
                        successful_indices = [i for i in successful_indices if successful_mask.get(i, False)]
        
        # Collect success indices from datasize experiments
        if fam_label in ds_data:
            for ds_frac in DATA_SIZES:
                df = ds_data[fam_label].get(ds_frac)
                if df is not None and "success" in df.columns:
                    successful_mask = df["success"] == 1
                    if successful_indices is None:
                        successful_indices = successful_mask.index[successful_mask].to_list()
                    else:
                        successful_indices = [i for i in successful_indices if successful_mask.get(i, False)]
        
        # Filter all dataframes in this family to only successful indices
        if successful_indices is not None:
            if fam_label in data:
                for solver_key in SOLVER_KEYS:
                    if data[fam_label][solver_key] is not None:
                        data[fam_label][solver_key] = data[fam_label][solver_key].loc[successful_indices].reset_index(drop=True)
            
            if fam_label in ds_data:
                for ds_frac in DATA_SIZES:
                    if ds_data[fam_label][ds_frac] is not None:
                        ds_data[fam_label][ds_frac] = ds_data[fam_label][ds_frac].loc[successful_indices].reset_index(drop=True)


def mean_stat(df: pd.DataFrame | None, col: str) -> float:
    if df is None or col not in df.columns:
        return np.nan
    return df[col].mean()


def bootstrap_ci(df: pd.DataFrame | None, col: str,
                 n_boot: int = 2000, ci: float = 0.95,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Return (lower_err, upper_err) half-widths for a bootstrap CI on the mean."""
    if df is None or col not in df.columns:
        return (np.nan, np.nan)
    data = df[col].dropna().values
    if len(data) < 2:
        return (0.0, 0.0)
    if rng is None:
        rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    mean = data.mean()
    return (mean - lo, hi - mean)   # (lower half-width, upper half-width)


# ── Save helper ───────────────────────────────────────────────────────────────
def save(fig: plt.Figure, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Stacked grouped bar: ctot decomposed into
#          planning time | inference time | Gittins index time
#          with two separate legends (solver colours / time decomposition)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_stacked_ctot(data):
    # Save original data for feasibility calculation before filtering
    data_orig = copy.deepcopy(data)
    
    # Filter to successful instances only (for time metrics)
    data = copy.deepcopy(data)
    filter_to_successful_instances(data, {})
    
    n_solvers  = len(SOLVER_KEYS)
    bar_width  = 0.17
    group_gap  = 1.0
    rng        = np.random.default_rng(42)

    # Offset each solver bar symmetrically within its group
    offsets = np.linspace(-(n_solvers - 1) / 2,
                           (n_solvers - 1) / 2,
                           n_solvers) * bar_width
    # Single x position at the centre of each subplot
    x_center = np.array([0.0])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=False)

    for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes[:3])):
        for i, solver_key in enumerate(SOLVER_KEYS):
            df = data[fam_label][solver_key]
            x  = x_center[0] + offsets[i]
            print(f"  {fam_label:12s}, {solver_key}  {len(df) if df is not None else 0} successful instances" )
            planning   = mean_stat(df, "planning_time")
            inference  = mean_stat(df, "inferenceCtot")
            gittins_t  = mean_stat(df, "gittinsCtot")
            base_color = SOLVER_COLORS[solver_key]

            # Bootstrap CI on total ctot
            ci_lo, ci_hi = bootstrap_ci(df, "ctot", rng=rng)
            total        = mean_stat(df, "ctot")

            if solver_key == "ELS":
                ax.bar(x, planning, width=bar_width * 0.9,
                       color=base_color, alpha=0.88, zorder=3)
            else:
                ax.bar(x, planning, width=bar_width * 0.9,
                       color=base_color, alpha=0.88, zorder=3)
                ax.bar(x, inference, width=bar_width * 0.9,
                       bottom=planning,
                       color=COLOR_INFERENCE, alpha=0.88, zorder=3)
                ax.bar(x, gittins_t, width=bar_width * 0.9,
                       bottom=planning + inference,
                       color=COLOR_GITTINS, alpha=0.88, zorder=3)

            # Error bar on total height
            ax.errorbar(x, total,
                        yerr=[[ci_lo], [ci_hi]],
                        fmt="none", color="black",
                        capsize=4, capthick=1.2,
                        elinewidth=1.2, zorder=5)

        # ── Per-subplot formatting ────────────────────────────────────────────
        ax.set_title(fam_label, fontsize=12)
        ax.set_xticks([])          # no x ticks — family name is the title
        ax.set_xlim(-0.5, 0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Mean Total Solve Time (s)")
    
    # ── 4th subplot: Feasibility lines ────────────────────────────────────────
    ax_feas = axes[3]
    x_positions = np.arange(len(FAMILY_LABELS))
    for solver_key in SOLVER_KEYS:
        feasibilities = []
        for fam_label in FAMILY_LABELS:
            df = data_orig[fam_label][solver_key]
            feas = mean_stat(df, "success") * 100
            feasibilities.append(feas)
        ax_feas.plot(x_positions, feasibilities, marker='o', linewidth=2.5,
                     markersize=8, label=SOLVER_LABELS[solver_key],
                     color=SOLVER_COLORS[solver_key], alpha=0.85)
    
    ax_feas.set_xticks(x_positions)
    ax_feas.set_xticklabels(FAMILY_LABELS, fontsize=11)
    ax_feas.set_ylabel("Feasibility (%)", fontsize=11)
    ax_feas.set_title("Feasibility", fontsize=12)
    ax_feas.set_ylim(0, 105)
    ax_feas.axhline(100, linestyle=":", color="grey", linewidth=1, zorder=1)
    ax_feas.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax_feas.set_axisbelow(True)

    fig.suptitle(
        "Mean Total Solve Time by Solver and Problem Family\n"
        "(stacked: total invested compute | total inference time | total Gittins index computation time;"
        "  error bars: 95% bootstrap CI)",
        fontsize=12, y=1.02,
    )

    # ── Legend 1: solver colours (below left) ────────────────────────────────
    solver_handles = [
        mpatches.Patch(color=SOLVER_COLORS[key], alpha=0.88,
                       label=SOLVER_LABELS[key])
        for key in SOLVER_KEYS
    ]
    legend1 = fig.legend(
        handles=solver_handles,
        title="Solver / Training Set",
        loc="lower left",
        bbox_to_anchor=(0.01, -0.18),
        framealpha=0.9,
        fontsize=9.5,
        title_fontsize=10,
        ncol=2,
    )

    # ── Legend 2: time decomposition (below right) ───────────────────────────
    decomp_handles = [
        mpatches.Patch(color=SOLVER_COLORS["rb2bTrain"], alpha=0.88,
                       label="Total Invested Compute"),
        mpatches.Patch(color=COLOR_INFERENCE, alpha=0.88,
                       label="Total Inference Time"),
        mpatches.Patch(color=COLOR_GITTINS, alpha=0.88,
                       label="Total Gittins Index Computation Time"),
    ]
    fig.legend(
        handles=decomp_handles,
        title="Time Decomposition",
        loc="lower right",
        bbox_to_anchor=(0.99, -0.18),
        framealpha=0.9,
        fontsize=9.5,
        title_fontsize=10,
        ncol=3,
    )

    fig.tight_layout()
    save(fig, "plot1_stacked_ctot_by_family.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Same stacked bar, but Gittins bars are the 5 data-fraction models
#          (averaged over 10 seeds each); ELS remains as baseline
# ═══════════════════════════════════════════════════════════════════════════════
def plot_stacked_ctot_datasizes(els_data, ds_data):
    """els_data  = load_all()   (only ELS key used)
       ds_data   = load_all_datasizes()
    """
    # Save original data for feasibility calculation before filtering
    els_data_orig = copy.deepcopy(els_data)
    ds_data_orig = copy.deepcopy(ds_data)
    
    # Filter to successful instances only
    els_data = copy.deepcopy(els_data)
    ds_data = copy.deepcopy(ds_data)
    filter_to_successful_instances(els_data, ds_data)
    
    # Bars: ELS + one per data fraction
    bar_keys   = ["ELS"] + DATA_SIZES          # mixed str / float
    n_bars     = len(bar_keys)
    bar_width  = 0.13
    rng        = np.random.default_rng(42)

    offsets = np.linspace(-(n_bars - 1) / 2,
                           (n_bars - 1) / 2,
                           n_bars) * bar_width

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.5), sharey=False)

    for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes[:3])):
        for i, key in enumerate(bar_keys):
            x = offsets[i]
            if key == "ELS":
                df         = els_data[fam_label]["ELS"]
                print(f"  {fam_label:12s}, {key}  {len(df) if df is not None else 0} successful instances" )
                # Datasize experiments were evaluated on the first 20 configs only
                if df is not None:
                    df = df.iloc[:20]
                base_color = SOLVER_COLORS["ELS"]
                planning   = mean_stat(df, "planning_time")
                total      = mean_stat(df, "ctot")
                ci_lo, ci_hi = bootstrap_ci(df, "ctot", rng=rng)
                ax.bar(x, planning, width=bar_width * 0.9,
                       color=base_color, alpha=0.88, zorder=3)
            else:
                df         = ds_data[fam_label][key]
                print(f"  {fam_label:12s}, {key}  {len(df) if df is not None else 0} successful instances" )
                base_color = DATASIZE_COLORS[key]
                planning   = mean_stat(df, "planning_time")
                inference  = mean_stat(df, "inferenceCtot")
                gittins_t  = mean_stat(df, "gittinsCtot")
                total      = mean_stat(df, "ctot")
                ci_lo, ci_hi = bootstrap_ci(df, "ctot", rng=rng)
                ax.bar(x, planning, width=bar_width * 0.9,
                       color=base_color, alpha=0.88, zorder=3)
                ax.bar(x, inference, width=bar_width * 0.9,
                       bottom=planning,
                       color=COLOR_INFERENCE, alpha=0.88, zorder=3)
                ax.bar(x, gittins_t, width=bar_width * 0.9,
                       bottom=planning + inference,
                       color=COLOR_GITTINS, alpha=0.88, zorder=3)

            ax.errorbar(x, total,
                        yerr=[[ci_lo], [ci_hi]],
                        fmt="none", color="black",
                        capsize=4, capthick=1.2,
                        elinewidth=1.2, zorder=5)

        ax.set_title(fam_label, fontsize=12)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Mean Total Solve Time (s)")
    
    # ── 4th subplot: Feasibility lines ────────────────────────────────────────
    ax_feas = axes[3]
    x_positions = np.arange(len(FAMILY_LABELS))
    
    # ELS line
    els_feasibilities = []
    for fam_label in FAMILY_LABELS:
        df = els_data_orig[fam_label]["ELS"]
        if df is not None:
            df = df.iloc[:20]
        feas = mean_stat(df, "success") * 100
        els_feasibilities.append(feas)
    ax_feas.plot(x_positions, els_feasibilities, marker='s', linewidth=2.5,
                 markersize=8, label="ELS", color=SOLVER_COLORS["ELS"], alpha=0.85)
    
    # Data size lines
    for ds_frac in DATA_SIZES:
        ds_feasibilities = []
        for fam_label in FAMILY_LABELS:
            df = ds_data_orig[fam_label][ds_frac]
            feas = mean_stat(df, "success") * 100
            ds_feasibilities.append(feas)
        ax_feas.plot(x_positions, ds_feasibilities, marker='o', linewidth=2.5,
                     markersize=8, label=DATASIZE_LABELS[ds_frac],
                     color=DATASIZE_COLORS[ds_frac], alpha=0.85)
    
    ax_feas.set_xticks(x_positions)
    ax_feas.set_xticklabels(FAMILY_LABELS, fontsize=11)
    ax_feas.set_ylabel("Feasibility (%)", fontsize=11)
    ax_feas.set_title("Feasibility", fontsize=12)
    ax_feas.set_ylim(0, 105)
    ax_feas.axhline(100, linestyle=":", color="grey", linewidth=1, zorder=1)
    ax_feas.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax_feas.set_axisbelow(True)

    fig.suptitle(
        "Mean Total Solve Time by Training Data Fraction and Problem Family\n"
        "(stacked: total invested compute | total inference time | total Gittins index computation time;"
        "  error bars: 95% bootstrap CI)",
        fontsize=12, y=1.02,
    )

    # ── Legend 1: solver/datasize colours ────────────────────────────────────
    solver_handles = [
        mpatches.Patch(color=SOLVER_COLORS["ELS"], alpha=0.88, label="ELS"),
    ] + [
        mpatches.Patch(color=DATASIZE_COLORS[ds], alpha=0.88,
                       label=DATASIZE_LABELS[ds])
        for ds in DATA_SIZES
    ]
    fig.legend(
        handles=solver_handles,
        title="Solver / Training Data Fraction",
        loc="lower left",
        bbox_to_anchor=(0.01, -0.18),
        framealpha=0.9,
        fontsize=9.5,
        title_fontsize=10,
        ncol=3,
    )

    # ── Legend 2: time decomposition ─────────────────────────────────────────
    decomp_handles = [
        mpatches.Patch(color=DATASIZE_COLORS[0.6], alpha=0.88,
                       label="Total Invested Compute"),
        mpatches.Patch(color=COLOR_INFERENCE, alpha=0.88,
                       label="Total Inference Time"),
        mpatches.Patch(color=COLOR_GITTINS, alpha=0.88,
                       label="Total Gittins Index Computation Time"),
    ]
    fig.legend(
        handles=decomp_handles,
        title="Time Decomposition",
        loc="lower right",
        bbox_to_anchor=(0.99, -0.18),
        framealpha=0.9,
        fontsize=9.5,
        title_fontsize=10,
        ncol=3,
    )

    fig.tight_layout()
    save(fig, "plot2_stacked_ctot_datasizes.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Line plots: ctot (total) and planning time vs data fraction
#          ELS drawn as a constant horizontal line
#          Bootstrap CI shown as shaded fill_between
# ═══════════════════════════════════════════════════════════════════════════════
def plot_line_datasizes(els_data, ds_data):
    """Two figures (3a = total ctot, 3b = planning time only), each with one
    subplot per problem family.  ELS appears as a horizontal constant line."""

    LINE_COLOR  = "#2171b5"   # Gittins line colour
    FILL_ALPHA  = 0.20
    ELS_COLOR   = "#d62728"
    ELS_ALPHA   = 0.15

    metrics = [
        ("ctot",          "Mean Total Solve Time — ctot  (s)",
         "plot3a_line_ctot_total.png",
         "Total Solve Time (ctot) vs Training Data Fraction"),
        ("planning_time",  "Mean Planning Time — ctot − metaCtot  (s)",
         "plot3b_line_planning_time.png",
         "Planning Time (ctot − metaCtot) vs Training Data Fraction"),
    ]

    rng = np.random.default_rng(42)

    for metric, ylabel, fname, title in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

        for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes)):

            # ── Gittins line over data fractions ─────────────────────────────
            means, lo_errs, hi_errs = [], [], []
            for ds in DATA_SIZES:
                df = ds_data[fam_label][ds]
                means.append(mean_stat(df, metric))
                ci_lo, ci_hi = bootstrap_ci(df, metric, rng=rng)
                lo_errs.append(ci_lo)
                hi_errs.append(ci_hi)

            means    = np.array(means)
            lo_errs  = np.array(lo_errs)
            hi_errs  = np.array(hi_errs)

            ax.plot(DATA_SIZES, means, marker="o", color=LINE_COLOR,
                    linewidth=2, zorder=4, label="Gittins (data fraction)")
            ax.fill_between(DATA_SIZES, means - lo_errs, means + hi_errs,
                            color=LINE_COLOR, alpha=FILL_ALPHA, zorder=3)

            # ── ELS constant line ─────────────────────────────────────────────
            els_df = els_data[fam_label]["ELS"]
            if els_df is not None:
                els_df = els_df.iloc[:20]
            els_mean         = mean_stat(els_df, metric)
            els_ci_lo, els_ci_hi = bootstrap_ci(els_df, metric, rng=rng)

            ax.axhline(els_mean, color=ELS_COLOR, linewidth=2,
                       linestyle="--", zorder=4, label="ELS (baseline)")
            ax.axhspan(els_mean - els_ci_lo, els_mean + els_ci_hi,
                       color=ELS_COLOR, alpha=ELS_ALPHA, zorder=3)

            # ── Formatting ────────────────────────────────────────────────────
            ax.set_title(fam_label, fontsize=12)
            ax.set_xlabel("Training Data Fraction")
            ax.set_xticks(DATA_SIZES)
            ax.set_xticklabels([f"{int(d*100)}%" for d in DATA_SIZES])
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(ylabel)
            if j == 2:
                ax.legend(fontsize=9.5, framealpha=0.9)

        fig.suptitle(title + "\n(shaded: 95% bootstrap CI)",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        save(fig, fname)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 – Box plots: ctot (total) and planning time
#          One box per solver/data-fraction, one subplot per family
# ═══════════════════════════════════════════════════════════════════════════════
def plot_boxplot_datasizes(els_data, ds_data):
    """Two figures (4a = total ctot, 4b = planning time), each with one
    subplot per problem family.  ELS is the first box, then one per data fraction."""

    metrics = [
        ("ctot",         "Total Solve Time — ctot  (s)",
         "plot4a_boxplot_ctot_total.png",
         "Distribution of Total Solve Time (ctot) by Training Data Fraction"),
        ("planning_time", "Planning Time — ctot − metaCtot  (s)",
         "plot4b_boxplot_planning_time.png",
         "Distribution of Planning Time (ctot − metaCtot) by Training Data Fraction"),
    ]

    bar_keys    = ["ELS"] + DATA_SIZES
    tick_labels = ["ELS"] + [f"{int(d*100)}%" for d in DATA_SIZES]
    box_colors  = [SOLVER_COLORS["ELS"]] + [DATASIZE_COLORS[ds] for ds in DATA_SIZES]
    positions   = list(range(len(bar_keys)))

    for metric, ylabel, fname, title in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)

        for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes)):
            groups = []
            for key in bar_keys:
                if key == "ELS":
                    df = els_data[fam_label]["ELS"]
                    if df is not None:
                        df = df.iloc[:20]
                else:
                    df = ds_data[fam_label][key]
                if df is not None and metric in df.columns:
                    groups.append(df[metric].dropna().values)
                else:
                    groups.append(np.array([]))

            bp = ax.boxplot(
                groups,
                positions=positions,
                patch_artist=True,
                widths=0.55,
                medianprops={"color": "black", "linewidth": 2},
                flierprops={"marker": ".", "markersize": 4, "alpha": 0.5,
                             "markeredgecolor": "grey"},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            ax.set_title(fam_label, fontsize=12)
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=9.5)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(ylabel)

        # shared legend
        handles = [mpatches.Patch(color=box_colors[0], alpha=0.75, label="ELS")] + [
            mpatches.Patch(color=box_colors[i + 1], alpha=0.75,
                           label=DATASIZE_LABELS[ds])
            for i, ds in enumerate(DATA_SIZES)
        ]
        fig.legend(
            handles=handles,
            title="Solver / Training Data Fraction",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            framealpha=0.9,
            fontsize=9.5,
            title_fontsize=10,
            ncol=len(bar_keys),
        )

        fig.suptitle(title, fontsize=12, y=1.02)
        fig.tight_layout()
        save(fig, fname)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 – Box plots for the trainset comparison (ELS vs rb2bTrain vs
#          rb2b3bTrain vs TunedGittins), two versions: ctot and planning time
# ═══════════════════════════════════════════════════════════════════════════════
def plot_boxplot_trainsets(data):
    metrics = [
        ("ctot",          "Total Solve Time — ctot  (s)",
         "plot5a_boxplot_trainsets_ctot.png",
         "Distribution of Total Solve Time (ctot) by Training Set"),
        ("planning_time",  "Planning Time — ctot − metaCtot  (s)",
         "plot5b_boxplot_trainsets_planning_time.png",
         "Distribution of Planning Time (ctot − metaCtot) by Training Set"),
    ]

    tick_labels = ["ELS", "RB(2,2,2)\nTrain Set",
                   "RB(2,2,2)+RB(3,3,2)\nTrain Set", "Full\nTrain Set"]
    box_colors  = [SOLVER_COLORS[k] for k in SOLVER_KEYS]
    positions   = list(range(len(SOLVER_KEYS)))

    for metric, ylabel, fname, title in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)

        for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes)):
            groups = []
            for key in SOLVER_KEYS:
                df = data[fam_label][key]
                if df is not None and metric in df.columns:
                    groups.append(df[metric].dropna().values)
                else:
                    groups.append(np.array([]))

            bp = ax.boxplot(
                groups,
                positions=positions,
                patch_artist=True,
                widths=0.55,
                medianprops={"color": "black", "linewidth": 2},
                flierprops={"marker": ".", "markersize": 4, "alpha": 0.5,
                            "markeredgecolor": "grey"},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            ax.set_title(fam_label, fontsize=12)
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(ylabel)

        handles = [
            mpatches.Patch(color=box_colors[i], alpha=0.75,
                           label=SOLVER_LABELS[key])
            for i, key in enumerate(SOLVER_KEYS)
        ]
        fig.legend(
            handles=handles,
            title="Solver / Training Set",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            framealpha=0.9,
            fontsize=9.5,
            title_fontsize=10,
            ncol=len(SOLVER_KEYS),
        )

        fig.suptitle(title, fontsize=12, y=1.02)
        fig.tight_layout()
        save(fig, fname)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 – Meta-reasoning time decomposition (Gittins vs Inference)
#   3 regions per subplot:
#     Region 1 (pos 0-1): rb2bTrain, rb2b3bTrain – first 20 rows (fair comparison)
#     Region 2 (pos 3-6): data fractions 20%–80%
#     Region 3 (pos 9):   Full Train Set (TunedGittins) – all rows
# ═══════════════════════════════════════════════════════════════════════════════
def plot_meta_decomposition(data, ds_data):
    rng = np.random.default_rng(42)

    # ── Layout ────────────────────────────────────────────────────────────────
    partial_solvers = ["rb2bTrain", "rb2b3bTrain"]   # named sets (excl. Full)
    partial_pos     = [0, 1]
    ds_sizes_shown  = [0.2, 0.4, 0.6, 0.8]          # 20%–80%, no 100%
    ds_pos          = [2.5, 3.5, 4.5, 5.5]
    full_pos        = [7.0]
    all_pos         = partial_pos + ds_pos + full_pos
    bar_width       = 0.75

    partial_labels = ["RB(2,2,2)\nTrain Set",
                      "RB(2,2,2)+\nRB(3,3,2)\nTrain Set"]
    ds_tick_labels = [f"{int(ds*100)}%\ndata" for ds in ds_sizes_shown]
    full_label     = ["Full\nTrain Set"]
    all_labels     = partial_labels + ds_tick_labels + full_label

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=False)

    def _plot_bar(ax, pos, df):
        g_mean   = mean_stat(df, "gittinsCtot")
        inf_mean = mean_stat(df, "inferenceCtot")
        ci_lo_g,   ci_hi_g   = bootstrap_ci(df, "gittinsCtot",   rng=rng)
        ci_lo_inf, ci_hi_inf = bootstrap_ci(df, "inferenceCtot", rng=rng)
        ax.bar(pos, g_mean,   width=bar_width,
               color=COLOR_GITTINS,   alpha=0.85, zorder=3)
        ax.bar(pos, inf_mean, width=bar_width, bottom=g_mean,
               color=COLOR_INFERENCE, alpha=0.85, zorder=3)
        total     = g_mean + inf_mean
        ci_lo_tot = np.sqrt(ci_lo_g**2  + ci_lo_inf**2)
        ci_hi_tot = np.sqrt(ci_hi_g**2  + ci_hi_inf**2)
        ax.errorbar(pos, total, yerr=[[ci_lo_tot], [ci_hi_tot]],
                    fmt="none", color="black",
                    capsize=4, capthick=1.2, elinewidth=1.2, zorder=5)

    for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes)):
        # ── Region 1: named partial train sets (first 20 rows) ────────────────
        for pos, key in zip(partial_pos, partial_solvers):
            df = data[fam_label][key].iloc[:20]
            _plot_bar(ax, pos, df)

        # ── Region 2: data fractions 20%–80% ─────────────────────────────────
        for pos, ds in zip(ds_pos, ds_sizes_shown):
            _plot_bar(ax, pos, ds_data[fam_label][ds])

        # ── Region 3: Full Train Set (100% data fraction, datasize experiment) ──
        _plot_bar(ax, full_pos[0], ds_data[fam_label][1.0])

        # ── Separators ────────────────────────────────────────────────────────
        ax.axvline(1.9, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(6.3, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_xticks(all_pos)
        ax.set_xticklabels(all_labels, fontsize=8.5)
        ax.set_title(fam_label, fontsize=12, pad=48)
        ax.set_xlim(-0.7, 7.7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Mean Meta-Reasoning Time (s)")

        # ── Group header annotations (x=data coords, y=axes fraction) ────────
        trans = ax.get_xaxis_transform()
        ax.text(0.5,  1.05, "Family-Specific\nTrain Sets",
                transform=trans, ha="center", va="bottom",
                fontsize=8, color="dimgrey", style="italic")
        ax.text(4.0,  1.05, "Training Data Fraction",
                transform=trans, ha="center", va="bottom",
                fontsize=8, color="dimgrey", style="italic")
        ax.text(7.0,  1.05, "Full Train Set",
                transform=trans, ha="center", va="bottom",
                fontsize=8, color="dimgrey", style="italic")

    fig.suptitle(
        "Meta-Reasoning Time Decomposition by Training Configuration and Problem Family\n"
        "(error bars: 95% bootstrap CI)",
        fontsize=12, y=1.04,
    )
    decomp_handles = [
        mpatches.Patch(color=COLOR_GITTINS,   alpha=0.85, label="Gittins Index Time  (gittinsCtot)"),
        mpatches.Patch(color=COLOR_INFERENCE, alpha=0.85, label="Inference Time  (inferenceCtot)"),
    ]
    fig.legend(handles=decomp_handles, title="Time Component",
               loc="lower center", bbox_to_anchor=(0.5, -0.12),
               framealpha=0.9, fontsize=9.5, title_fontsize=10, ncol=2)
    fig.tight_layout()
    save(fig, "plot6_meta_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7 – Success rate by training set (one subplot per family)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_success_trainsets(data):
    rng       = np.random.default_rng(42)
    n_solvers = len(SOLVER_KEYS)
    bar_width = 0.18
    offsets   = np.linspace(-(n_solvers - 1) / 2,
                             (n_solvers - 1) / 2,
                             n_solvers) * bar_width

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)

    for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes)):
        for i, key in enumerate(SOLVER_KEYS):
            df    = data[fam_label][key]
            x     = offsets[i]
            color = SOLVER_COLORS[key]
            rate  = mean_stat(df, "success") * 100
            ci_lo, ci_hi = bootstrap_ci(df, "success", rng=rng)

            ax.bar(x, rate, width=bar_width * 0.9,
                   color=color, alpha=0.88, zorder=3)
            ax.errorbar(x, rate,
                        yerr=[[ci_lo * 100], [ci_hi * 100]],
                        fmt="none", color="black",
                        capsize=4, capthick=1.2, elinewidth=1.2, zorder=5)

        tick_labels = ["ELS", "RB(2,2,2)\nTrain", "RB(2,2,2)+\nRB(3,3,2)", "Full\nTrain"]
        ax.set_xticks(offsets)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(fam_label, fontsize=12)
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0, 110)
        ax.axhline(100, linestyle=":", color="grey", linewidth=1, zorder=1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Success Rate (%)")

    fig.suptitle(
        "Success Rate by Training Set and Problem Family\n"
        "(error bars: 95% bootstrap CI)",
        fontsize=12, y=1.02,
    )
    handles = [mpatches.Patch(color=SOLVER_COLORS[k], alpha=0.88, label=SOLVER_LABELS[k])
               for k in SOLVER_KEYS]
    fig.legend(handles=handles, title="Solver / Training Set",
               loc="lower center", bbox_to_anchor=(0.5, -0.12),
               framealpha=0.9, fontsize=9.5, title_fontsize=10, ncol=len(SOLVER_KEYS))
    fig.tight_layout()
    save(fig, "plot7_success_trainsets.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 8 – Success rate by data fraction (one subplot per family)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_success_datasizes(els_data, ds_data):
    rng       = np.random.default_rng(42)
    bar_keys  = ["ELS"] + DATA_SIZES
    n_bars    = len(bar_keys)
    bar_width = 0.13
    offsets   = np.linspace(-(n_bars - 1) / 2,
                             (n_bars - 1) / 2,
                             n_bars) * bar_width
    box_colors = [SOLVER_COLORS["ELS"]] + [DATASIZE_COLORS[ds] for ds in DATA_SIZES]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)

    for j, (fam_label, ax) in enumerate(zip(FAMILY_LABELS, axes)):
        for i, key in enumerate(bar_keys):
            x     = offsets[i]
            color = box_colors[i]

            if key == "ELS":
                df = els_data[fam_label]["ELS"]
                if df is not None:
                    df = df.iloc[:20]
            else:
                df = ds_data[fam_label][key]

            rate = mean_stat(df, "success") * 100
            ci_lo, ci_hi = bootstrap_ci(df, "success", rng=rng)

            ax.bar(x, rate, width=bar_width * 0.9,
                   color=color, alpha=0.88, zorder=3)
            ax.errorbar(x, rate,
                        yerr=[[ci_lo * 100], [ci_hi * 100]],
                        fmt="none", color="black",
                        capsize=4, capthick=1.2, elinewidth=1.2, zorder=5)

        tick_labels = ["ELS"] + [f"{int(d*100)}%" for d in DATA_SIZES]
        ax.set_xticks(offsets)
        ax.set_xticklabels(tick_labels, fontsize=9.5)
        ax.set_title(fam_label, fontsize=12)
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0, 110)
        ax.axhline(100, linestyle=":", color="grey", linewidth=1, zorder=1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Success Rate (%)")
    
    # ── 4th subplot: Feasibility lines ────────────────────────────────────────
    ax_feas = axes[3]
    x_positions = np.arange(len(FAMILY_LABELS))
    
    # ELS line
    els_feasibilities = []
    for fam_label in FAMILY_LABELS:
        df = els_data_orig[fam_label]["ELS"]
        if df is not None:
            df = df.iloc[:20]
        feas = mean_stat(df, "success") * 100
        els_feasibilities.append(feas)
    ax_feas.plot(x_positions, els_feasibilities, marker='s', linewidth=2.5,
                 markersize=8, label="ELS", color=SOLVER_COLORS["ELS"], alpha=0.85)
    
    # Data size lines
    for ds_frac in DATA_SIZES:
        ds_feasibilities = []
        for fam_label in FAMILY_LABELS:
            df = ds_data_orig[fam_label][ds_frac]
            feas = mean_stat(df, "success") * 100
            ds_feasibilities.append(feas)
        ax_feas.plot(x_positions, ds_feasibilities, marker='o', linewidth=2.5,
                     markersize=8, label=DATASIZE_LABELS[ds_frac],
                     color=DATASIZE_COLORS[ds_frac], alpha=0.85)
    
    ax_feas.set_xticks(x_positions)
    ax_feas.set_xticklabels(FAMILY_LABELS, fontsize=11)
    ax_feas.set_ylabel("Feasibility (%)", fontsize=11)
    ax_feas.set_title("Feasibility", fontsize=12)
    ax_feas.set_ylim(0, 105)
    ax_feas.axhline(100, linestyle=":", color="grey", linewidth=1, zorder=1)
    ax_feas.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax_feas.set_axisbelow(True)

    fig.suptitle(
        "Success Rate by Training Data Fraction and Problem Family\n"
        "(error bars: 95% bootstrap CI)",
        fontsize=12, y=1.02,
    )
    handles = [mpatches.Patch(color=box_colors[0], alpha=0.88, label="ELS")] + [
        mpatches.Patch(color=box_colors[i + 1], alpha=0.88, label=DATASIZE_LABELS[ds])
        for i, ds in enumerate(DATA_SIZES)
    ]
    fig.legend(handles=handles, title="Solver / Training Data Fraction",
               loc="lower center", bbox_to_anchor=(0.5, -0.12),
               framealpha=0.9, fontsize=9.5, title_fontsize=10, ncol=n_bars)
    fig.tight_layout()
    save(fig, "plot2_success_datasizes.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Print feasibility table
# ═══════════════════════════════════════════════════════════════════════════════
def print_feasibility_table(data, els_data, ds_data):
    print("\n" + "=" * 72)
    print("FEASIBILITY / SUCCESS RATE  (mean % over all problems in the file)")
    print("=" * 72)

    # ── By training set ───────────────────────────────────────────────────────
    print("\n── By Training Set ──")
    col_w  = 30
    header = f"{'Family':<14}" + "".join(f"{SOLVER_LABELS[k]:>{col_w}}" for k in SOLVER_KEYS)
    print(header)
    print("-" * len(header))
    for fam_label in FAMILY_LABELS:
        row = f"{fam_label:<14}"
        for key in SOLVER_KEYS:
            df = data[fam_label][key]
            df_use = df.iloc[:20] if (key == "ELS" and df is not None) else df
            val = mean_stat(df_use, "success") * 100
            row += f"{val:>{col_w}.1f}%"
        print(row)

    # ── By data fraction ─────────────────────────────────────────────────────
    print("\n── By Training Data Fraction ──")
    ds_labels = ["ELS (20 cfg)"] + [f"{int(d*100)}% data" for d in DATA_SIZES]
    col_w2  = 14
    header2 = f"{'Family':<14}" + "".join(f"{lbl:>{col_w2}}" for lbl in ds_labels)
    print(header2)
    print("-" * len(header2))
    for fam_label in FAMILY_LABELS:
        row = f"{fam_label:<14}"
        els_df = els_data[fam_label]["ELS"]
        if els_df is not None:
            els_df = els_df.iloc[:20]
        row += f"{mean_stat(els_df, 'success') * 100:>{col_w2}.1f}%"
        for ds in DATA_SIZES:
            df  = ds_data[fam_label][ds]
            row += f"{mean_stat(df, 'success') * 100:>{col_w2}.1f}%"
        print(row)
    print("=" * 72 + "\n")
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading comparison data...")
    data = load_all()

    print("\nLoading datasize data...")
    ds_data = load_all_datasizes()

    print("\nGenerating plots...")
    plot_stacked_ctot(data)
    plot_stacked_ctot_datasizes(data, ds_data)
    plot_line_datasizes(data, ds_data)
    plot_boxplot_datasizes(data, ds_data)
    plot_boxplot_trainsets(data)
    plot_meta_decomposition(data, ds_data)
    plot_success_trainsets(data)
    # plot_success_datasizes(data, ds_data)
    print_feasibility_table(data, data, ds_data)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
