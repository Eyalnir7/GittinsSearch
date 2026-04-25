"""
evaluate_models.py

Evaluates trained GNN models against the ground-truth Gittins index
computed from empirical feasibility/time data.

Usage
-----
python evaluate_models.py \
    --data_dir  /path/to/dataset/split/test \
    --models_dir /path/to/models_folder \
    [--output    results.csv] \
    [--quantile_levels 0.1 0.3 0.5 0.7 0.9] \
    [--beta 0.999]

Expected layout of --data_dir
------------------------------
  aggregated_configurations.json
  aggregated_waypoints.csv          (for WAYPOINTS models)
  aggregated_rrt_by_action.csv      (for RRT models)
  aggregated_lgp_by_plan.csv        (for LGP models)

Expected layout of --models_dir
--------------------------------
  best_constraint_gnn_model_<run_id>.pt
  model_meta_<run_id>.json          (for every .pt file)

The script:
  1. Discovers all (model, metadata) pairs via run_id suffix matching.
  2. For each model, loads the appropriate dataset CSV.
  3. Computes the ground-truth Gittins index per row from empirical data.
  4. Runs inference and converts predictions to a Gittins index using
     ground-truth values for the components the model does NOT predict:
       - QUANTILE_REGRESSION_FEAS  : predicted feas-quantiles + GT infeas-quantiles + GT avgFeas
       - QUANTILE_REGRESSION_INFEAS: GT feas-quantiles + predicted infeas-quantiles + GT avgFeas
       - FEASIBILITY               : GT feas-quantiles + GT infeas-quantiles + predicted avgFeas
  5. Reports Kendall-tau, Spearman-rho, and MSE/MAE metrics.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr, beta as scipy_beta, norm as scipy_norm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure the Learning package is importable when running from any cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # evaluation/
_LEARNING_DIR = _SCRIPT_DIR.parent                     # Learning/
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

from bandit_process import BanditProcess, MarkovChain, BanditType   # noqa: E402
from bestScriptedModel import ScriptableConstraintGNN, forward_heteroBatch  # noqa: E402
from dataset import HeteroGraphDataset, parse_malformed_plan          # noqa: E402
from enums import TaskType, NodeType                                  # noqa: E402

# torch_geometric DataLoader
from torch_geometric.loader import DataLoader                         # noqa: E402


# ===========================================================================
# Constants
# ===========================================================================
QUANTILE_LEVELS_DEFAULT = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA_DEFAULT = 0.999

NODE_TYPE_TO_CSV = {
    NodeType.WAYPOINTS: "aggregated_waypoints.csv",
    NodeType.RRT:       "aggregated_rrt_by_action.csv",
    NodeType.LGP:       "aggregated_lgp_by_plan.csv",
}

# fallback name the user mentioned for LGP
_LGP_ALT_CSV = "aggregated_lgp_by_action.csv"


# ===========================================================================
# Ground-truth cache helpers
# ===========================================================================
def _gt_cache_path(csv_path: Path, quantile_levels: list) -> Path:
    """Return the path of the pickle cache file for this CSV + quantile combo."""
    q_str = "_".join(str(int(q * 100)) for q in sorted(quantile_levels))
    return csv_path.parent / f"{csv_path.stem}_gt_cache_q{q_str}.pkl"


def _load_gt_cache(cache_path: Path) -> dict | None:
    if cache_path.exists():
        print(f"  Loading GT cache from {cache_path.name} …")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def _save_gt_cache(cache_path: Path, payload: dict) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  GT cache saved to {cache_path.name}")


# ===========================================================================
# Helper: locate dataset CSV
# ===========================================================================
def _csv_for_node_type(data_dir: Path, node_type: NodeType) -> Path:
    primary = data_dir / NODE_TYPE_TO_CSV[node_type]
    if primary.exists():
        return primary
    if node_type == NodeType.LGP:
        alt = data_dir / _LGP_ALT_CSV
        if alt.exists():
            return alt
    raise FileNotFoundError(
        f"Cannot find CSV for {node_type} in {data_dir}. "
        f"Expected '{NODE_TYPE_TO_CSV[node_type]}' (or '{_LGP_ALT_CSV}' for LGP)."
    )


# ===========================================================================
# Helper: parse run_id from a filename
# The run_id is the last underscore-separated token before the extension.
# ===========================================================================
def _run_id_from_filename(name: str) -> str:
    stem = Path(name).stem           # strips extension
    return stem.rsplit("_", 1)[-1]


# ===========================================================================
# Metric function (mirrors the notebook)
# ===========================================================================
def ordering_change_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Compute ordering-change metrics comparing arrays *a* (ground truth)
    and *b* (predictions).
    """
    if len(a) != len(b):
        raise ValueError("Arrays must have the same length.")
    N = len(a)

    ranks_a = np.argsort(np.argsort(a))
    ranks_b = np.argsort(np.argsort(b))

    tau, _ = kendalltau(a, b)
    total_pairs = N * (N - 1) / 2
    discordant_pairs = (1 - tau) / 2 * total_pairs
    kendall_distance = discordant_pairs / total_pairs

    rho, _ = spearmanr(a, b)
    rank_diffs = ranks_a - ranks_b
    spearman_rank_distance = float(np.sum(rank_diffs ** 2))

    avg_sq_diff = float(np.mean((a - b) ** 2))
    avg_abs_diff = float(np.mean(np.abs(a - b)))

    return {
        "kendall_tau":             float(tau),
        "kendall_distance":        float(kendall_distance),
        "spearman_rho":            float(rho),
        "spearman_rank_distance":  spearman_rank_distance,
        "avg_squared_difference":  avg_sq_diff,
        "avg_absolute_difference": avg_abs_diff,
    }


# ===========================================================================
# Ground-truth chain extraction (from the notebook)
# ===========================================================================
def get_chain_probs(feas_array, time_array):
    """
    Returns (done_transitions, done_times, fail_transitions, fail_times)
    suitable for passing to MarkovChain / BanditProcess.
    """
    feas_array = np.array(feas_array)
    time_array = np.array(time_array)

    feas_times   = time_array[feas_array == 1]
    infeas_times = time_array[feas_array == 0]

    feas_values,   feas_counts   = np.unique(feas_times,   return_counts=True)
    infeas_values, infeas_counts = np.unique(infeas_times, return_counts=True)

    all_values = np.union1d(feas_values, infeas_values)
    feas_counts_full   = np.zeros(len(all_values), dtype=float)
    infeas_counts_full = np.zeros(len(all_values), dtype=float)

    for i, v in enumerate(all_values):
        if v in feas_values:
            feas_counts_full[i]   = feas_counts[feas_values == v][0]
        if v in infeas_values:
            infeas_counts_full[i] = infeas_counts[infeas_values == v][0]

    all_counts      = feas_counts_full + infeas_counts_full
    cum_sum_counts  = np.cumsum(all_counts[::-1])[::-1]

    done_arr = feas_counts_full   / cum_sum_counts
    fail_arr = infeas_counts_full / cum_sum_counts

    # prepend t=0 row with p=0
    done_arr   = np.insert(done_arr, 0, 0.0)
    fail_arr   = np.insert(fail_arr, 0, 0.0)
    all_values = np.insert(all_values, 0, 0)

    return done_arr, all_values, fail_arr, all_values


def compute_gittins_index_from_chain(done_transitions, done_times,
                                     fail_transitions, fail_times):
    mc = MarkovChain(
        done_transitions=done_transitions,
        done_times=done_times,
        fail_transitions=fail_transitions,
        fail_times=fail_times,
        type=BanditType.LINE,
    )
    bp = BanditProcess(markov_chains=[mc], bandit_types=[BanditType.LINE])
    gi, tau = bp.get_gittins_index()
    return gi, tau


def calculate_gittins_index(row) -> float:
    done_trans, done_times, fail_trans, fail_times = get_chain_probs(
        row["feas"], row["time"]
    )
    gi, _ = compute_gittins_index_from_chain(
        done_trans, done_times, fail_trans, fail_times
    )
    return gi


# ===========================================================================
# Quantile-based Gittins computation (from the notebook)
# ===========================================================================
def get_chain_probs_from_quantile_values(
    feas_quantiles, infeas_quantiles, quantile_levels, avgFeas
):
    """
    Convert quantile predictions into (done_trans, feas_q, fail_trans, infeas_q)
    suitable for MarkovChain.
    """
    feas_quantiles   = np.asarray(feas_quantiles,   dtype=float)
    infeas_quantiles = np.asarray(infeas_quantiles, dtype=float)

    unique_quantiles = np.sort(np.unique(
        np.concatenate((feas_quantiles, infeas_quantiles))
    ))

    in_feas_q   = [q in feas_quantiles   for q in unique_quantiles]
    in_infeas_q = [q in infeas_quantiles for q in unique_quantiles]

    done_trans = []
    fail_trans = []
    sum_done   = 0.0
    sum_fail   = 0.0
    next_transition = 1.0
    done_index = 0
    fail_index = 0
    last_quantile = -1

    for i, Aq in enumerate(unique_quantiles):
        current_done = 0.0
        current_fail = 0.0

        if in_feas_q[i]:
            if Aq == last_quantile:
                done_trans.pop()
            qi = quantile_levels[done_index]
            current_done = float(np.clip(
                (avgFeas * qi - sum_done) / next_transition, 0, 1
            ))
            sum_done += current_done
            done_index += 1
            done_trans.append(current_done)

        if in_infeas_q[i]:
            if Aq == last_quantile:
                fail_trans.pop()
            qi = quantile_levels[fail_index]
            current_fail = float(np.clip(
                ((1 - avgFeas) * qi - sum_fail) / next_transition, 0, 1
            ))
            sum_fail += current_fail
            fail_index += 1
            fail_trans.append(current_fail)

        next_transition *= (1.0 - current_done - current_fail)
        last_quantile = Aq

    # Normalise boundary
    if len(fail_trans) > 0:
        if in_feas_q[-1] and not in_infeas_q[-1]:
            fail_trans.append(1.0 - done_trans[-1])
            infeas_quantiles = np.append(infeas_quantiles, unique_quantiles[-1])
        elif not in_feas_q[-1] and in_infeas_q[-1]:
            fail_trans[-1] = 1.0
        elif in_feas_q[-1] and in_infeas_q[-1]:
            fail_trans[-1] = 1.0 - done_trans[-1]
    elif len(done_trans) > 0:
        done_trans[-1] = 1.0

    feas_quantiles   = np.sort(np.unique(feas_quantiles))
    infeas_quantiles = np.sort(np.unique(infeas_quantiles))
    return done_trans, feas_quantiles, fail_trans, infeas_quantiles


def compute_gittins_from_quantiles(
    feas_quantiles, infeas_quantiles, quantile_levels, avgFeas
) -> float:
    """
    Returns Gittins index from quantile-based chain representation.
    Returns NaN on any error.
    """
    try:
        feas_q   = np.asarray(feas_quantiles, dtype=float)
        infeas_q = np.asarray(infeas_quantiles, dtype=float)

        if np.any(np.isnan(feas_q)) or np.any(np.isnan(infeas_q)):
            return float("nan")

        done_trans, feas_q_out, fail_trans, infeas_q_out = \
            get_chain_probs_from_quantile_values(feas_q, infeas_q, quantile_levels, avgFeas)

        done_arr = np.array(done_trans, dtype=float)
        fail_arr = np.array(fail_trans, dtype=float)

        if np.any(np.isnan(done_arr)) or np.any(np.isnan(fail_arr)):
            return float("nan")

        mc = MarkovChain(done_arr, np.array(feas_q_out), fail_arr, np.array(infeas_q_out), BanditType.LINE)
        bp = BanditProcess([mc], [BanditType.LINE])
        gi, _ = bp.get_gittins_index()
        return float(gi)
    except Exception as e:
        warnings.warn(f"Gittins computation failed: {e}")
        return float("nan")


# ===========================================================================
# Ground-truth quantile extraction from empirical data
# ===========================================================================
def compute_gt_quantiles(row, quantile_levels):
    """Returns (feas_quantiles_gt, infeas_quantiles_gt) as int arrays."""
    feas_arr = np.array(row["feas"])
    time_arr = np.array(row["time"])

    feas_times   = time_arr[feas_arr == 1]
    infeas_times = time_arr[feas_arr == 0]

    if len(feas_times) > 0:
        fq = np.ceil(np.percentile(feas_times,   [q * 100 for q in quantile_levels])).astype(int)
    else:
        fq = np.array([], dtype=int)

    if len(infeas_times) > 0:
        iq = np.ceil(np.percentile(infeas_times, [q * 100 for q in quantile_levels])).astype(int)
    else:
        iq = np.array([], dtype=int)

    return fq, iq


# ===========================================================================
# Model loading
# ===========================================================================
def load_model(model_path: Path, output_dim: int, device: torch.device) -> ScriptableConstraintGNN:
    model = ScriptableConstraintGNN(output_dim=output_dim)
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    # Support both raw state-dicts and checkpoint dicts (with 'model_state_dict' key)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # Drop keys that no longer exist in the model (e.g. removed-but-unused modules)
    model_keys = set(model.state_dict().keys())
    unexpected = [k for k in list(state_dict.keys()) if k not in model_keys]
    for k in unexpected:
        del state_dict[k]
    if unexpected:
        print(f"  [INFO] Dropped unused keys from checkpoint: {unexpected}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def output_dim_for_task(task_type: TaskType, num_quantiles: int = 5) -> int:
    if task_type == TaskType.FEASIBILITY:
        return 1
    return num_quantiles


# ===========================================================================
# Inference helpers (mirrors the notebook)
# ===========================================================================
def _normalize_plan(plan_val):
    if plan_val is None:
        return None
    if torch.is_tensor(plan_val):
        plan_val = plan_val.tolist()
    if isinstance(plan_val, (list, tuple)):
        return json.loads(json.dumps(plan_val))
    if isinstance(plan_val, str):
        stripped = plan_val.strip()
        try:
            return parse_malformed_plan(stripped)
        except Exception:
            try:
                return json.loads(stripped)
            except Exception:
                return stripped
    return plan_val


def _make_key(file_id, plan_val):
    norm = _normalize_plan(plan_val)
    try:
        plan_str = json.dumps(norm, separators=(",", ":"))
    except TypeError:
        plan_str = str(norm)
    return (file_id, plan_str)


def _make_key_rrt(file_id, plan_val, action_num):
    norm = _normalize_plan(plan_val)
    try:
        plan_str = json.dumps(norm, separators=(",", ":"))
    except TypeError:
        plan_str = str(norm)
    return (file_id, plan_str, action_num)


def _scalar_from_tensor(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        v = v.view(-1).tolist()
    if isinstance(v, (list, tuple)):
        return v[0] if len(v) else None
    return v


def _extract_key(batch, node_type: NodeType):
    file_id  = _scalar_from_tensor(getattr(batch, "file_id",   None))
    plan_val = _scalar_from_tensor(getattr(batch, "taskPlan",  None))
    action_n = _scalar_from_tensor(getattr(batch, "actionNum", None))

    # fallback: look in per-node attributes
    if file_id is None or plan_val is None:
        for nt in batch.node_types:
            nd = batch[nt]
            if file_id  is None and hasattr(nd, "file_id"):
                file_id  = _scalar_from_tensor(nd.file_id)
            if plan_val is None and hasattr(nd, "taskPlan"):
                plan_val = _scalar_from_tensor(nd.taskPlan)
            if (node_type == NodeType.RRT) and action_n is None and hasattr(nd, "actionNum"):
                action_n = _scalar_from_tensor(nd.actionNum)
            if file_id is not None and plan_val is not None:
                if node_type != NodeType.RRT or action_n is not None:
                    break

    if node_type == NodeType.RRT:
        return _make_key_rrt(file_id, plan_val, action_n)
    return _make_key(file_id, plan_val)


def run_inference(model, dataloader, device, task_type: TaskType,
                  node_type: NodeType, num_quantiles: int = 5) -> dict:
    """
    Returns a dict mapping canonical key -> prediction array.
    For FEASIBILITY: scalar probability.
    For QUANTILE_REGRESSION_*: np.ndarray of quantile values.
    """
    model.eval()
    predictions: dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  inference", leave=False):
            batch = batch.to(device)
            logits = forward_heteroBatch(model, batch)

            if task_type == TaskType.FEASIBILITY:
                prob = torch.sigmoid(logits).item()
                key  = _extract_key(batch, node_type)
                if key is not None:
                    predictions[key] = prob
            else:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                base = logits[:, 0].unsqueeze(-1)
                preds = [base]
                cumulative = base
                for i in range(1, num_quantiles):
                    inc = torch.nn.functional.softplus(logits[:, i].unsqueeze(-1))
                    cumulative = cumulative + inc
                    preds.append(cumulative)
                pred_quantiles = torch.cat(preds, dim=-1).squeeze(0).cpu().numpy()
                key = _extract_key(batch, node_type)
                if key is not None:
                    predictions[key] = pred_quantiles

    return predictions


# ===========================================================================
# Build canonical key from a dataframe row (mirrors notebook helpers)
# ===========================================================================
def _row_key(row, node_type: NodeType):
    if node_type == NodeType.RRT:
        return _make_key_rrt(row["file_id"], row.get("plan"), row.get("actionNum"))
    return _make_key(row["file_id"], row.get("plan"))


# ===========================================================================
# Main evaluation for one (model, dataset) pair
# ===========================================================================
def evaluate_single_model(
    model_path: Path,
    metadata: dict,
    data_dir: Path,
    quantile_levels: list,
    device: torch.device,
) -> dict:
    """
    Evaluate one model and return a results dict with metrics and metadata.
    """
    run_id    = metadata["run_id"]
    task_str  = metadata["task"]          # e.g. "QUANTILE_REGRESSION_FEAS"
    node_str  = metadata["node_type"]     # e.g. "WAYPOINTS"

    task_type = TaskType[task_str]
    node_type = NodeType[node_str]
    num_q     = len(quantile_levels)

    print(f"\n{'='*60}")
    print(f"Model:     {model_path.name}")
    print(f"Run ID:    {run_id}")
    print(f"Task:      {task_str}   Node: {node_str}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load the dataset CSV
    # ------------------------------------------------------------------
    csv_path = _csv_for_node_type(data_dir, node_type)
    print(f"Dataset:   {csv_path}")

    converters = {
        "feas": ast.literal_eval,
        "time": ast.literal_eval,
        "plan": parse_malformed_plan,
    }
    df = pd.read_csv(str(csv_path), converters=converters)
    df["feas"] = df["feas"].apply(lambda x: [int(i) for i in x])
    df["time"] = df["time"].apply(lambda x: [float(i) for i in x])
    for idx in range(len(df)):
        df.at[idx, "time"] = [int(np.ceil(t / 0.01)) for t in df.at[idx, "time"]]

    print(f"Rows:      {len(df)}")

    # ------------------------------------------------------------------
    # 2. Compute (or load cached) ground-truth quantities
    # ------------------------------------------------------------------
    df["avgFeas"] = df["feas"].apply(lambda x: sum(x) / len(x) if len(x) else 0.0)

    cache_path = _gt_cache_path(csv_path, quantile_levels)
    cache = _load_gt_cache(cache_path)

    if cache is not None and len(cache.get("gittins_index_gt", [])) == len(df):
        df["gittins_index_gt"] = cache["gittins_index_gt"]
        df["feas_q_gt"]        = cache["feas_q_gt"]
        df["infeas_q_gt"]      = cache["infeas_q_gt"]
        print("Ground-truth quantities loaded from cache.")
    else:
        print("Computing ground-truth Gittins indices...")
        gt_gi = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  GT Gittins", leave=False):
            try:
                gi = calculate_gittins_index(row)
            except Exception as e:
                warnings.warn(f"GT Gittins failed for row: {e}")
                gi = float("nan")
            gt_gi.append(gi)
        df["gittins_index_gt"] = gt_gi

        # Ground-truth quantiles (needed when model only predicts one component)
        print("Computing ground-truth quantiles...")
        gt_fq, gt_iq = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  GT quantiles", leave=False):
            fq, iq = compute_gt_quantiles(row, quantile_levels)
            gt_fq.append(fq)
            gt_iq.append(iq)
        df["feas_q_gt"]   = gt_fq
        df["infeas_q_gt"] = gt_iq

        _save_gt_cache(cache_path, {
            "gittins_index_gt": df["gittins_index_gt"].tolist(),
            "feas_q_gt":        df["feas_q_gt"].tolist(),
            "infeas_q_gt":      df["infeas_q_gt"].tolist(),
        })

    # ------------------------------------------------------------------
    # 3. Build HeteroGraphDataset for inference
    # ------------------------------------------------------------------
    print("Building graph dataset for inference...")
    data_dir_str = str(data_dir) + "/"   # HeteroGraphDataset expects trailing /
    output_dim   = output_dim_for_task(task_type, num_q)

    dataset = HeteroGraphDataset(
        input_path=data_dir_str,
        nodeType=node_type,
        taskType=task_type,
        device=torch.device("cpu"),  # build on CPU; batches are moved to device in run_inference
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 4. Load model and run inference
    # ------------------------------------------------------------------
    print(f"Loading model (output_dim={output_dim})...")
    if metadata.get("scripted"):
        model = load_scripted_model(model_path, device)
    else:
        model = load_model(model_path, output_dim, device)

    print("Running inference...")
    predictions = run_inference(model, loader, device, task_type, node_type, num_q)
    print(f"  Predictions collected: {len(predictions)}")

    # ------------------------------------------------------------------
    # 5. Attach predictions to dataframe rows via canonical keys
    # ------------------------------------------------------------------
    def get_pred(row):
        key = _row_key(row, node_type)
        return predictions.get(key, None)

    df["_pred"] = df.apply(get_pred, axis=1)
    matched = df["_pred"].notna().sum()
    print(f"  Matched {matched}/{len(df)} rows to predictions")

    # ------------------------------------------------------------------
    # 6. Compute predicted Gittins using model output + GT for other components
    # ------------------------------------------------------------------
    print("Computing predicted Gittins indices...")
    pred_gi = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Pred Gittins", leave=False):
        pred = row["_pred"]
        if pred is None or (isinstance(pred, float) and np.isnan(pred)):
            pred_gi.append(float("nan"))
            continue

        avg_feas   = float(row["avgFeas"])
        feas_q_gt  = row["feas_q_gt"]
        infeas_q_gt = row["infeas_q_gt"]

        try:
            if task_type == TaskType.FEASIBILITY:
                # Use predicted avgFeas, GT quantiles
                predicted_avg_feas = float(pred)
                gi = compute_gittins_from_quantiles(
                    feas_q_gt, infeas_q_gt, quantile_levels, predicted_avg_feas
                )

            elif task_type == TaskType.QUANTILE_REGRESSION_FEAS:
                pred_arr = np.ceil(np.asarray(pred, dtype=float)).astype(int)
                if node_type == NodeType.RRT:
                    # For RRT: avgFeas is always 1, no infeas component
                    gi = compute_gittins_from_quantiles(
                        pred_arr, np.array([], dtype=int), quantile_levels, 1.0
                    )
                else:
                    gi = compute_gittins_from_quantiles(
                        pred_arr, infeas_q_gt, quantile_levels, avg_feas
                    )

            elif task_type == TaskType.QUANTILE_REGRESSION_INFEAS:
                pred_arr = np.ceil(np.asarray(pred, dtype=float)).astype(int)
                gi = compute_gittins_from_quantiles(
                    feas_q_gt, pred_arr, quantile_levels, avg_feas
                )
            else:
                gi = float("nan")
        except Exception as e:
            warnings.warn(f"Pred Gittins failed: {e}")
            gi = float("nan")

        pred_gi.append(gi)

    df["gittins_index_pred"] = pred_gi

    # ------------------------------------------------------------------
    # 7. Filter out rows with NaN in either GT or pred Gittins
    # ------------------------------------------------------------------
    valid = df["gittins_index_gt"].notna() & df["gittins_index_pred"].notna()
    n_valid = valid.sum()
    n_total = len(df)
    print(f"  Valid rows for metric computation: {n_valid}/{n_total}")

    if n_valid < 2:
        print("  WARNING: Not enough valid rows to compute metrics.")
        metrics = {k: float("nan") for k in [
            "kendall_tau", "kendall_distance", "spearman_rho",
            "spearman_rank_distance", "avg_squared_difference", "avg_absolute_difference"
        ]}
    else:
        gt_arr   = df.loc[valid, "gittins_index_gt"].values.astype(float)
        pred_arr = df.loc[valid, "gittins_index_pred"].values.astype(float)
        metrics  = ordering_change_metrics(gt_arr, pred_arr)

    # ------------------------------------------------------------------
    # 8. Print metrics summary
    # ------------------------------------------------------------------
    print("\n  ----- METRICS -----")
    for k, v in metrics.items():
        print(f"  {k:<35s}: {v:.6f}")
    print("  -------------------")

    return {
        "run_id":        run_id,
        "model_path":    str(model_path),
        "task":          task_str,
        "node_type":     node_str,
        "dataset":       str(csv_path),
        "n_total":       n_total,
        "n_valid":       n_valid,
        "n_matched":     int(matched),
        **metrics,
    }


# ===========================================================================
# RRT evaluation: only QUANTILE_REGRESSION_FEAS, with avgFeas=1 / no infeas
# ===========================================================================
def evaluate_rrt_model(
    model_path: Path,
    metadata: dict,
    data_dir: Path,
    quantile_levels: list,
    device: torch.device,
) -> dict:
    """
    Evaluate the QUANTILE_REGRESSION_FEAS model for the RRT node type.

    RRT actions always succeed, so the Gittins chain is constructed with:
      - predicted feas quantiles  (from the model)
      - avgFeas = 1.0             (hardcoded – no GT value used)
      - infeas quantiles = []     (hardcoded – no GT value used)
    """
    run_id   = metadata["run_id"]
    node_str = "RRT"
    node_type = NodeType.RRT
    task_type = TaskType.QUANTILE_REGRESSION_FEAS
    num_q     = len(quantile_levels)

    print(f"\n{'='*60}")
    print(f"RRT evaluation  —  QUANTILE_REGRESSION_FEAS (avgFeas=1, no infeas)")
    print(f"Model:  {model_path.name}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    csv_path = _csv_for_node_type(data_dir, node_type)
    print(f"Dataset:   {csv_path}")

    converters = {
        "feas": ast.literal_eval,
        "time": ast.literal_eval,
        "plan": parse_malformed_plan,
    }
    df = pd.read_csv(str(csv_path), converters=converters)
    df["feas"] = df["feas"].apply(lambda x: [int(i) for i in x])
    df["time"] = df["time"].apply(lambda x: [float(i) for i in x])
    for idx in range(len(df)):
        df.at[idx, "time"] = [int(np.ceil(t / 0.01)) for t in df.at[idx, "time"]]
    print(f"Rows:      {len(df)}")

    # ------------------------------------------------------------------
    # 2. Compute (or load cached) ground-truth Gittins from the exact chain
    # ------------------------------------------------------------------
    cache_path = _gt_cache_path(csv_path, quantile_levels)
    cache = _load_gt_cache(cache_path)

    if cache is not None and len(cache.get("gittins_index_gt", [])) == len(df):
        df["gittins_index_gt"] = cache["gittins_index_gt"]
        print("Ground-truth Gittins loaded from cache.")
    else:
        print("Computing ground-truth Gittins indices...")
        gt_gi = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  GT Gittins", leave=False):
            try:
                gi = calculate_gittins_index(row)
            except Exception as e:
                warnings.warn(f"GT Gittins failed: {e}")
                gi = float("nan")
            gt_gi.append(gi)
        df["gittins_index_gt"] = gt_gi
        # Merge into existing cache if present, else create new
        cache_payload = dict(cache) if cache is not None else {}
        cache_payload["gittins_index_gt"] = df["gittins_index_gt"].tolist()
        _save_gt_cache(cache_path, cache_payload)

    # ------------------------------------------------------------------
    # 3. Build dataset / run inference
    # ------------------------------------------------------------------
    print("Building graph dataset for inference...")
    dataset = HeteroGraphDataset(
        input_path=str(data_dir) + "/",
        nodeType=node_type,
        taskType=task_type,
        device=torch.device("cpu"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Loading model (output_dim={num_q})...")
    if metadata.get("scripted"):
        model = load_scripted_model(model_path, device)
    else:
        model = load_model(model_path, num_q, device)

    print("Running inference...")
    predictions = run_inference(model, loader, device, task_type, node_type, num_q)
    print(f"  Predictions collected: {len(predictions)}")

    def get_pred(row):
        return predictions.get(_row_key(row, node_type), None)

    df["_pred"] = df.apply(get_pred, axis=1)
    matched = df["_pred"].notna().sum()
    print(f"  Matched {matched}/{len(df)} rows to predictions")

    # ------------------------------------------------------------------
    # 4. Compute predicted Gittins: pred_fq + avgFeas=1 + no infeas
    # ------------------------------------------------------------------
    print("Computing predicted Gittins indices (avgFeas=1, no infeas)...")
    pred_gi = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Pred Gittins", leave=False):
        pred = row["_pred"]
        if pred is None or (isinstance(pred, float) and np.isnan(pred)):
            pred_gi.append(float("nan"))
            continue
        try:
            pred_fq = np.ceil(np.asarray(pred, dtype=float)).astype(int)
            gi = compute_gittins_from_quantiles(
                pred_fq,
                np.array([], dtype=int),  # no infeas component
                quantile_levels,
                1.0,                      # avgFeas always 1 for RRT
            )
        except Exception as e:
            warnings.warn(f"Pred Gittins failed: {e}")
            gi = float("nan")
        pred_gi.append(gi)

    df["gittins_index_pred"] = pred_gi

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    valid   = df["gittins_index_gt"].notna() & df["gittins_index_pred"].notna()
    n_valid = valid.sum()
    n_total = len(df)
    print(f"  Valid rows: {n_valid}/{n_total}")

    if n_valid < 2:
        metrics = {k: float("nan") for k in [
            "kendall_tau", "kendall_distance", "spearman_rho",
            "spearman_rank_distance", "avg_squared_difference", "avg_absolute_difference"
        ]}
    else:
        gt_arr   = df.loc[valid, "gittins_index_gt"].values.astype(float)
        pred_arr = df.loc[valid, "gittins_index_pred"].values.astype(float)
        metrics  = ordering_change_metrics(gt_arr, pred_arr)

    print("\n  ----- METRICS (RRT) -----")
    for k, v in metrics.items():
        print(f"  {k:<35s}: {v:.6f}")
    print("  -------------------------")

    return {
        "run_id":    run_id,
        "model_path": str(model_path),
        "task":      "QUANTILE_REGRESSION_FEAS",
        "node_type": node_str,
        "dataset":   str(csv_path),
        "n_total":   n_total,
        "n_valid":   int(n_valid),
        "n_matched": int(matched),
        **metrics,
    }


# ===========================================================================
# Triplet evaluation: all three models for one node_type predict together
# ===========================================================================
def evaluate_triplet_models(
    models: dict,   # task_str -> (Path, metadata_dict)
    node_str: str,
    data_dir: Path,
    quantile_levels: list,
    device: torch.device,
) -> dict:
    """
    Evaluate the combined performance of all three models for a given node_type.
    - FEASIBILITY          -> predicted avgFeas
    - QUANTILE_REGRESSION_FEAS   -> predicted feas quantiles
    - QUANTILE_REGRESSION_INFEAS -> predicted infeas quantiles
    All three predictions are combined to compute the Gittins index.
    """
    node_type = NodeType[node_str]
    num_q     = len(quantile_levels)

    run_ids = "|".join(meta["run_id"] for _, meta in models.values())
    print(f"\n{'='*60}")
    print(f"TRIPLET evaluation  —  node_type: {node_str}")
    print(f"Run IDs: {run_ids}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    csv_path = _csv_for_node_type(data_dir, node_type)
    print(f"Dataset:   {csv_path}")

    converters = {
        "feas": ast.literal_eval,
        "time": ast.literal_eval,
        "plan": parse_malformed_plan,
    }
    df = pd.read_csv(str(csv_path), converters=converters)
    df["feas"] = df["feas"].apply(lambda x: [int(i) for i in x])
    df["time"] = df["time"].apply(lambda x: [float(i) for i in x])
    for idx in range(len(df)):
        df.at[idx, "time"] = [int(np.ceil(t / 0.01)) for t in df.at[idx, "time"]]
    print(f"Rows:      {len(df)}")

    # ------------------------------------------------------------------
    # 2. Ground-truth (load cache or compute)
    # ------------------------------------------------------------------
    df["avgFeas"] = df["feas"].apply(lambda x: sum(x) / len(x) if len(x) else 0.0)

    cache_path = _gt_cache_path(csv_path, quantile_levels)
    cache = _load_gt_cache(cache_path)

    if cache is not None and len(cache.get("gittins_index_gt", [])) == len(df):
        df["gittins_index_gt"] = cache["gittins_index_gt"]
        print("Ground-truth Gittins loaded from cache.")
    else:
        print("Computing ground-truth Gittins indices...")
        gt_gi = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  GT Gittins", leave=False):
            try:
                gi = calculate_gittins_index(row)
            except Exception as e:
                warnings.warn(f"GT Gittins failed: {e}")
                gi = float("nan")
            gt_gi.append(gi)
        df["gittins_index_gt"] = gt_gi

    # ------------------------------------------------------------------
    # 3. Run inference for each of the three models
    # ------------------------------------------------------------------
    all_preds: dict[str, dict] = {}   # task_str -> key->prediction
    for task_str, (model_path, meta) in models.items():
        task_type  = TaskType[task_str]
        output_dim = output_dim_for_task(task_type, num_q)
        print(f"  Loading {task_str} model ({model_path.name})...")
        if meta.get("scripted"):
            model = load_scripted_model(model_path, device)
        else:
            model = load_model(model_path, output_dim, device)

        # Build dataset/loader for this node_type
        dataset = HeteroGraphDataset(
            input_path=str(data_dir) + "/",
            nodeType=node_type,
            taskType=task_type,
            device=torch.device("cpu"),  # build on CPU; batches are moved to device in run_inference
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"  Running inference ({task_str})...")
        all_preds[task_str] = run_inference(model, loader, device, task_type, node_type, num_q)
        print(f"    Predictions: {len(all_preds[task_str])}")

    # ------------------------------------------------------------------
    # 4. Combine predictions and compute Gittins
    # ------------------------------------------------------------------
    print("Computing combined predicted Gittins indices...")
    pred_gi = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Combined Gittins", leave=False):
        key = _row_key(row, node_type)
        try:
            feas_pred_raw  = all_preds["FEASIBILITY"].get(key)
            pred_fq_raw    = all_preds["QUANTILE_REGRESSION_FEAS"].get(key)
            pred_iq_raw    = all_preds["QUANTILE_REGRESSION_INFEAS"].get(key)

            if feas_pred_raw is None or pred_fq_raw is None or pred_iq_raw is None:
                pred_gi.append(float("nan"))
                continue

            pred_fq = np.ceil(np.asarray(pred_fq_raw, dtype=float)).astype(int)
            pred_iq = np.ceil(np.asarray(pred_iq_raw, dtype=float)).astype(int)
            pred_avg_feas = float(feas_pred_raw)   # already a probability from run_inference

            gi = compute_gittins_from_quantiles(
                pred_fq, pred_iq, quantile_levels, pred_avg_feas
            )
        except Exception as e:
            warnings.warn(f"Triplet Gittins failed: {e}")
            gi = float("nan")
        pred_gi.append(gi)

    df["gittins_index_pred"] = pred_gi

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    valid   = df["gittins_index_gt"].notna() & df["gittins_index_pred"].notna()
    n_valid = valid.sum()
    n_total = len(df)
    print(f"  Valid rows: {n_valid}/{n_total}")

    if n_valid < 2:
        metrics = {k: float("nan") for k in [
            "kendall_tau", "kendall_distance", "spearman_rho",
            "spearman_rank_distance", "avg_squared_difference", "avg_absolute_difference"
        ]}
    else:
        gt_arr   = df.loc[valid, "gittins_index_gt"].values.astype(float)
        pred_arr = df.loc[valid, "gittins_index_pred"].values.astype(float)
        metrics  = ordering_change_metrics(gt_arr, pred_arr)

    print("\n  ----- METRICS (TRIPLET) -----")
    for k, v in metrics.items():
        print(f"  {k:<35s}: {v:.6f}")
    print("  -----------------------------")

    return {
        "run_id":    run_ids,
        "model_path": "|".join(str(p) for p, _ in models.values()),
        "task":      "TRIPLET",
        "node_type": node_str,
        "dataset":   str(csv_path),
        "n_total":   n_total,
        "n_valid":   n_valid,
        "n_matched": int(valid.sum()),
        **metrics,
    }


# ===========================================================================
# Approximation-method evaluation (no models)
# ===========================================================================
def evaluate_approximation_method(
    csv_path: Path,
    node_str: str,
    quantile_levels: list,
) -> dict:
    """
    Evaluate the quantile-chain approximation method itself, with no model involved.

    Ground-truth quantiles (feas/infeas) and avgFeas are derived directly from
    the empirical data. The resulting quantile-chain Gittins index is compared
    against the exact ground-truth Gittins index, giving a ceiling on how well
    any model can do given this approximation scheme.
    """
    node_type = NodeType[node_str]
    print(f"\n{'='*60}")
    print(f"APPROXIMATION evaluation  —  node_type: {node_str}")
    print(f"Dataset:   {csv_path}")
    print(f"{'='*60}")

    converters = {
        "feas": ast.literal_eval,
        "time": ast.literal_eval,
        "plan": parse_malformed_plan,
    }
    df = pd.read_csv(str(csv_path), converters=converters)
    df["feas"] = df["feas"].apply(lambda x: [int(i) for i in x])
    df["time"] = df["time"].apply(lambda x: [float(i) for i in x])
    for idx in range(len(df)):
        df.at[idx, "time"] = [int(np.ceil(t / 0.01)) for t in df.at[idx, "time"]]
    print(f"Rows:      {len(df)}")

    df["avgFeas"] = df["feas"].apply(lambda x: sum(x) / len(x) if len(x) else 0.0)

    # ------------------------------------------------------------------
    # Load or compute ground-truth cache
    # ------------------------------------------------------------------
    cache_path = _gt_cache_path(csv_path, quantile_levels)
    cache = _load_gt_cache(cache_path)

    _cache_has_gi  = cache is not None and len(cache.get("gittins_index_gt", [])) == len(df)
    _cache_has_q   = cache is not None and "feas_q_gt" in cache and "infeas_q_gt" in cache

    if _cache_has_gi:
        df["gittins_index_gt"] = cache["gittins_index_gt"]
        if _cache_has_q:
            df["feas_q_gt"]    = cache["feas_q_gt"]
            df["infeas_q_gt"]  = cache["infeas_q_gt"]
            print("Ground-truth quantities loaded from cache.")
        else:
            print("Ground-truth Gittins loaded from cache (quantiles missing – computing now)...")
    else:
        print("Computing ground-truth Gittins indices...")
        gt_gi = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  GT Gittins", leave=False):
            try:
                gi = calculate_gittins_index(row)
            except Exception as e:
                warnings.warn(f"GT Gittins failed for row: {e}")
                gi = float("nan")
            gt_gi.append(gi)
        df["gittins_index_gt"] = gt_gi

    if not _cache_has_q:
        print("Computing ground-truth quantiles...")
        gt_fq, gt_iq = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  GT quantiles", leave=False):
            fq, iq = compute_gt_quantiles(row, quantile_levels)
            gt_fq.append(fq)
            gt_iq.append(iq)
        df["feas_q_gt"]   = gt_fq
        df["infeas_q_gt"] = gt_iq

        # Merge into / create the cache so it is complete for next time
        cache_payload = dict(cache) if cache is not None else {}
        cache_payload.update({
            "gittins_index_gt": df["gittins_index_gt"].tolist(),
            "feas_q_gt":        df["feas_q_gt"].tolist(),
            "infeas_q_gt":      df["infeas_q_gt"].tolist(),
        })
        _save_gt_cache(cache_path, cache_payload)

    # ------------------------------------------------------------------
    # Compute approximated Gittins from GT quantiles + GT avgFeas
    # ------------------------------------------------------------------
    print("Computing approximated Gittins indices from GT quantiles...")
    approx_gi = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Approx Gittins", leave=False):
        try:
            avg_feas     = float(row["avgFeas"])
            feas_q_gt    = row["feas_q_gt"]
            infeas_q_gt  = row["infeas_q_gt"]

            if node_type == NodeType.RRT:
                # RRT actions always succeed (avgFeas == 1), no infeas component
                gi = compute_gittins_from_quantiles(
                    feas_q_gt, np.array([], dtype=int), quantile_levels, 1.0
                )
            else:
                gi = compute_gittins_from_quantiles(
                    feas_q_gt, infeas_q_gt, quantile_levels, avg_feas
                )
        except Exception as e:
            warnings.warn(f"Approx Gittins failed: {e}")
            gi = float("nan")
        approx_gi.append(gi)

    df["gittins_index_approx"] = approx_gi

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    valid   = df["gittins_index_gt"].notna() & df["gittins_index_approx"].notna()
    n_valid = valid.sum()
    n_total = len(df)
    print(f"  Valid rows: {n_valid}/{n_total}")

    if n_valid < 2:
        metrics = {k: float("nan") for k in [
            "kendall_tau", "kendall_distance", "spearman_rho",
            "spearman_rank_distance", "avg_squared_difference", "avg_absolute_difference"
        ]}
    else:
        gt_arr   = df.loc[valid, "gittins_index_gt"].values.astype(float)
        pred_arr = df.loc[valid, "gittins_index_approx"].values.astype(float)
        metrics  = ordering_change_metrics(gt_arr, pred_arr)

    print("\n  ----- METRICS (APPROXIMATION) -----")
    for k, v in metrics.items():
        print(f"  {k:<35s}: {v:.6f}")
    print("  -----------------------------------")

    return {
        "run_id":     "GT_APPROX",
        "model_path": "N/A",
        "task":       "APPROXIMATION",
        "node_type":  node_str,
        "dataset":    str(csv_path),
        "n_total":    n_total,
        "n_valid":    int(n_valid),
        "n_matched":  int(n_valid),
        **metrics,
    }


# ===========================================================================
# Discover (model_path, metadata) pairs in a directory
# ===========================================================================
def discover_model_pairs(models_dir: Path) -> list[tuple[Path, dict]]:
    """
    Scan a directory for (*.pt, model_meta_*.json) pairs matched by run_id.
    Returns list of (pt_path, metadata_dict).
    """
    # Collect all metadata files
    meta_files = list(models_dir.glob("model_meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(
            f"No 'model_meta_*.json' files found in {models_dir}"
        )

    pairs = []
    for meta_path in sorted(meta_files):
        run_id = _run_id_from_filename(meta_path.name)

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Verify run_id consistency
        if metadata.get("run_id") and metadata["run_id"] != run_id:
            run_id = metadata["run_id"]   # trust metadata over filename

        # Find the corresponding .pt file (ends with _{run_id}.pt)
        matching_pts = list(models_dir.glob(f"*_{run_id}.pt"))
        if not matching_pts:
            print(f"  WARNING: No .pt file found for run_id='{run_id}' (from {meta_path.name}). Skipping.")
            continue
        if len(matching_pts) > 1:
            print(f"  WARNING: Multiple .pt files for run_id='{run_id}': {matching_pts}. Using first.")

        pairs.append((matching_pts[0], metadata))

    return pairs


# ===========================================================================
# Scripted-model discovery and loading
# ===========================================================================
_TASK_PREFIXES = [
    "QUANTILE_REGRESSION_FEAS",
    "QUANTILE_REGRESSION_INFEAS",
    "FEASIBILITY",
]
_NODE_TYPES = ["WAYPOINTS", "RRT", "LGP"]


def infer_metadata_from_scripted_filename(filename: str) -> dict | None:
    """
    Infer task and node_type from a scripted model filename.

    Expected pattern: model_<TASK>_<NODE_TYPE>[_...].pt
    e.g. model_FEASIBILITY_WAYPOINTS_p0.2_randomBlocks_all.pt
         model_QUANTILE_REGRESSION_FEAS_RRT_p0.2_randomBlocks_all.pt
    """
    stem = Path(filename).stem  # drop .pt extension
    name = stem.removeprefix("model_")

    # Match task (longest-first to avoid FEASIBILITY matching before QUANTILE_REGRESSION_FEAS)
    task = None
    for t in _TASK_PREFIXES:
        if name.startswith(t):
            task = t
            name = name[len(t):].lstrip("_")
            break

    if task is None:
        return None

    # Match node type
    node_type = None
    for nt in _NODE_TYPES:
        if name.startswith(nt):
            node_type = nt
            break

    if node_type is None:
        return None

    run_id = stem.removeprefix("model_")   # stable id derived from full stem
    return {"run_id": run_id, "task": task, "node_type": node_type, "scripted": True}


def discover_scripted_models(models_dir: Path) -> list[tuple[Path, dict]]:
    """
    Scan a directory for TorchScript model files (model_*.pt) and infer
    task / node_type metadata from their filenames.

    Returns list of (pt_path, metadata_dict).
    """
    pt_files = sorted(models_dir.glob("model_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No 'model_*.pt' files found in {models_dir}")

    pairs = []
    for pt_path in pt_files:
        metadata = infer_metadata_from_scripted_filename(pt_path.name)
        if metadata is None:
            print(f"  WARNING: Could not infer metadata from '{pt_path.name}'. Skipping.")
            continue
        pairs.append((pt_path, metadata))

    return pairs


def load_scripted_model(model_path: Path, device: torch.device):
    """Load a TorchScript model saved with torch.jit.save()."""
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


# ===========================================================================
# Datasize-experiment helpers
# ===========================================================================
def discover_datasize_structure(datasize_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Scan *datasize_dir* for the expected two-level layout::

        datasize_{p}/
            seed_{i}/       <- contains model files in --models_dir format
            seed_{j}/
            ...
        datasize_{q}/
            ...

    Returns ``{p_str: {seed_str: seed_models_path}}``.
    """
    result: dict[str, dict[str, Path]] = {}
    for ds_dir in sorted(datasize_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        m = re.match(r"^datasize_(.+)$", ds_dir.name)
        if not m:
            continue
        p_str = m.group(1)
        seeds: dict[str, Path] = {}
        for seed_dir in sorted(ds_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            sm = re.match(r"^seed_(.+)$", seed_dir.name)
            if not sm:
                continue
            seeds[sm.group(1)] = seed_dir
        if seeds:
            result[p_str] = seeds
    return result


def evaluate_datasize_all(
    datasize_dir: Path,
    data_dir: Path,
    quantile_levels: list,
    device: torch.device,
    scripted: bool,
    detail_output: Path,
) -> list[dict]:
    """
    Evaluate all (datasize, seed) model combinations as triplets.

    * Per-seed detailed results are written to *detail_output*.
    * Returns a list of aggregate-result dicts (one per datasize × node_type)
      with mean / median / std of Kendall's τ across seeds – intended for
      inclusion in the main results CSV.
    """
    TRIPLET_NODE_TYPES = ["WAYPOINTS", "LGP"]
    TRIPLET_TASKS = [
        "FEASIBILITY",
        "QUANTILE_REGRESSION_FEAS",
        "QUANTILE_REGRESSION_INFEAS",
    ]

    structure = discover_datasize_structure(datasize_dir)
    if not structure:
        print(f"WARNING: No 'datasize_*' directories found in {datasize_dir}")
        return []

    detail_rows: list[dict] = []

    for p_str, seeds in structure.items():
        print(f"\n{'#'*60}")
        print(f"# Datasize: {p_str}  ({len(seeds)} seeds)")
        print(f"{'#'*60}")

        for seed_str, seed_models_dir in seeds.items():
            print(f"\n  --- Seed: {seed_str}  ({seed_models_dir}) ---")

            # Discover models in this seed directory
            try:
                if scripted:
                    pairs = discover_scripted_models(seed_models_dir)
                else:
                    pairs = discover_model_pairs(seed_models_dir)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}. Skipping seed {seed_str}.")
                continue

            if not pairs:
                print(f"  WARNING: No models found in {seed_models_dir}. Skipping.")
                continue

            pair_index: dict[tuple, tuple] = {}
            for pt_path, metadata in pairs:
                key = (metadata.get("node_type"), metadata.get("task"))
                pair_index[key] = (pt_path, metadata)

            for node_str in TRIPLET_NODE_TYPES:
                if all((node_str, t) in pair_index for t in TRIPLET_TASKS):
                    models_for_triplet = {t: pair_index[(node_str, t)] for t in TRIPLET_TASKS}
                    try:
                        result = evaluate_triplet_models(
                            models=models_for_triplet,
                            node_str=node_str,
                            data_dir=data_dir,
                            quantile_levels=quantile_levels,
                            device=device,
                        )
                        result["datasize_p"] = p_str
                        result["seed"] = seed_str
                        detail_rows.append(result)
                    except Exception as e:
                        print(
                            f"\nERROR in triplet evaluation for {node_str} "
                            f"(datasize={p_str}, seed={seed_str}): {e}"
                        )
                        import traceback
                        traceback.print_exc()
                else:
                    missing = [t for t in TRIPLET_TASKS if (node_str, t) not in pair_index]
                    print(
                        f"\n[SKIP] Triplet for {node_str} "
                        f"(datasize={p_str}, seed={seed_str}): missing tasks {missing}"
                    )

            # RRT: only QUANTILE_REGRESSION_FEAS (avgFeas=1, no infeas component)
            rrt_key = ("RRT", "QUANTILE_REGRESSION_FEAS")
            if rrt_key in pair_index:
                pt_path, rrt_meta = pair_index[rrt_key]
                try:
                    result = evaluate_rrt_model(
                        model_path=pt_path,
                        metadata=rrt_meta,
                        data_dir=data_dir,
                        quantile_levels=quantile_levels,
                        device=device,
                    )
                    result["datasize_p"] = p_str
                    result["seed"] = seed_str
                    detail_rows.append(result)
                except Exception as e:
                    print(
                        f"\nERROR in RRT evaluation "
                        f"(datasize={p_str}, seed={seed_str}): {e}"
                    )
                    import traceback
                    traceback.print_exc()
            else:
                print(
                    f"\n[SKIP] RRT evaluation "
                    f"(datasize={p_str}, seed={seed_str}): no QUANTILE_REGRESSION_FEAS model found"
                )

    if not detail_rows:
        print("No datasize results collected.")
        return []

    # Save per-seed details
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(str(detail_output), index=False)
    print(f"\nDatasize detailed results saved to: {detail_output}")

    # Build aggregate rows (one per datasize × node_type)
    agg_rows: list[dict] = []
    for (p_str, node_str), group in detail_df.groupby(["datasize_p", "node_type"]):
        tau_vals = group["kendall_tau"].dropna()
        n_seeds  = len(group)
        agg_rows.append({
            "run_id":             f"DATASIZE_AGG_p{p_str}_{node_str}",
            "model_path":         f"datasize_{p_str}/{node_str}",
            "task":               "TRIPLET_DATASIZE_AGG",
            "node_type":          node_str,
            "datasize_p":         p_str,
            "n_seeds":            n_seeds,
            "dataset":            str(data_dir),
            "n_total":            float("nan"),
            "n_valid":            float("nan"),
            "n_matched":          float("nan"),
            "kendall_tau":        float("nan"),
            "kendall_tau_mean":   float(tau_vals.mean())   if len(tau_vals) else float("nan"),
            "kendall_tau_median": float(tau_vals.median()) if len(tau_vals) else float("nan"),
            "kendall_tau_std":    float(tau_vals.std())    if len(tau_vals) else float("nan"),
        })

    return agg_rows


# ===========================================================================
# CLI entry point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GNN models against ground-truth Gittins index."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing the dataset CSVs and aggregated_configurations.json.",
    )
    parser.add_argument(
        "--models_dir",
        nargs="+",
        default=None,
        help="One or more directories, each containing .pt model files and "
             "model_meta_*.json metadata files. The full evaluation is performed "
             "for every directory provided. "
             "Optional when --eval_approx is used without model evaluation.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.csv",
        help="Path to save the results CSV (default: evaluation_results.csv).",
    )
    parser.add_argument(
        "--quantile_levels",
        nargs="+",
        type=float,
        default=QUANTILE_LEVELS_DEFAULT,
        help="Quantile levels used by the quantile-regression models "
             "(default: 0.1 0.3 0.5 0.7 0.9).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda / mps / cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        default=False,
        help="Treat --models_dir as containing TorchScript models (model_*.pt) "
             "saved by script_model.py. Metadata (task, node_type) will be "
             "inferred from filenames instead of model_meta_*.json files.",
    )
    parser.add_argument(
        "--eval_approx",
        action="store_true",
        default=False,
        help="Also evaluate the quantile-chain approximation method itself (no models). "
             "For each available CSV in --data_dir, the Gittins index is computed from "
             "ground-truth quantiles and avgFeas and compared against the exact GT Gittins. "
             "Can be combined with model evaluation or used alone (without --models_dir).",
    )
    parser.add_argument(
        "--datasize_dir",
        default=None,
        help="Path to the datasize experiment directory. "
             "Expected layout: datasize_{p}/seed_{i}/<model files>. "
             "For each (datasize, seed) combination, the three task models per node type "
             "are evaluated as a triplet. Per-seed results are saved to a separate "
             "detail CSV (derived from --output), and per-datasize aggregate statistics "
             "(mean, median, std of Kendall's tau) are appended to the main output CSV.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    data_dir      = Path(args.data_dir).resolve()
    models_dirs   = [Path(p).resolve() for p in args.models_dir] if args.models_dir else []
    datasize_dir  = Path(args.datasize_dir).resolve() if args.datasize_dir else None
    output        = Path(args.output)
    q_levels      = sorted(args.quantile_levels)

    # Detail CSV for per-seed datasize results: <stem>_datasize_detail<suffix>
    datasize_detail_output = output.with_name(
        output.stem + "_datasize_detail" + output.suffix
    )

    if not data_dir.is_dir():
        sys.exit(f"Error: --data_dir '{data_dir}' is not a directory.")
    for _md in models_dirs:
        if not _md.is_dir():
            sys.exit(f"Error: --models_dir '{_md}' is not a directory.")
    if datasize_dir is not None and not datasize_dir.is_dir():
        sys.exit(f"Error: --datasize_dir '{datasize_dir}' is not a directory.")
    if not models_dirs and not args.eval_approx and datasize_dir is None:
        sys.exit(
            "Error: at least one of --models_dir, --datasize_dir, or --eval_approx "
            "must be provided."
        )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            # Verify that the GPU's compute capability is actually supported by
            # this PyTorch build before committing to CUDA (e.g. sm_61 GTX 1070
            # is detected but unsupported by PyTorch builds requiring sm_70+).
            try:
                torch.zeros(1).cuda()
                device = torch.device("cuda")
            except Exception as _cuda_err:
                print(f"WARNING: CUDA detected but unusable ({_cuda_err}). Falling back to CPU.")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
    print(f"Using device: {device}")
    print(f"Quantile levels: {q_levels}")

    all_results = []

    # ------------------------------------------------------------------
    # Discover and evaluate models (once per --models_dir entry)
    # ------------------------------------------------------------------
    TRIPLET_NODE_TYPES = ["WAYPOINTS", "LGP"]
    TRIPLET_TASKS = ["FEASIBILITY", "QUANTILE_REGRESSION_FEAS", "QUANTILE_REGRESSION_INFEAS"]

    for models_dir in models_dirs:
        print(f"\nScanning for models in: {models_dir}")
        if args.scripted:
            print("Mode: scripted TorchScript models (metadata inferred from filenames)")
            pairs = discover_scripted_models(models_dir)
        else:
            pairs = discover_model_pairs(models_dir)
        print(f"Found {len(pairs)} model(s) to evaluate.")

        if not pairs:
            print(f"WARNING: No models found in {models_dir}.")
            continue

        pair_index: dict[tuple, tuple] = {}
        for pt_path, metadata in pairs:
            key = (metadata.get("node_type"), metadata.get("task"))
            pair_index[key] = (pt_path, metadata)

        for node_str in TRIPLET_NODE_TYPES:
            if all((node_str, t) in pair_index for t in TRIPLET_TASKS):
                models_for_triplet = {t: pair_index[(node_str, t)] for t in TRIPLET_TASKS}
                try:
                    result = evaluate_triplet_models(
                        models=models_for_triplet,
                        node_str=node_str,
                        data_dir=data_dir,
                        quantile_levels=q_levels,
                        device=device,
                    )
                    result["models_dir"] = str(models_dir)
                    all_results.append(result)
                except Exception as e:
                    print(f"\nERROR in triplet evaluation for {node_str}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                missing_tasks = [t for t in TRIPLET_TASKS if (node_str, t) not in pair_index]
                print(f"\n[SKIP] Triplet for {node_str} in {models_dir.name}: missing models for {missing_tasks}")

        # RRT: only QUANTILE_REGRESSION_FEAS is meaningful (avgFeas=1, no infeas component)
        rrt_key = ("RRT", "QUANTILE_REGRESSION_FEAS")
        if rrt_key in pair_index:
            pt_path, rrt_meta = pair_index[rrt_key]
            try:
                result = evaluate_rrt_model(
                    model_path=pt_path,
                    metadata=rrt_meta,
                    data_dir=data_dir,
                    quantile_levels=q_levels,
                    device=device,
                )
                result["models_dir"] = str(models_dir)
                all_results.append(result)
            except Exception as e:
                print(f"\nERROR in RRT evaluation: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[SKIP] RRT evaluation in {models_dir.name}: no QUANTILE_REGRESSION_FEAS model found")

    # ------------------------------------------------------------------
    # Approximation-method evaluation (--eval_approx)
    # ------------------------------------------------------------------
    if args.eval_approx:
        print(f"\n{'='*60}")
        print("Running approximation-method evaluation (no models)...")
        print(f"{'='*60}")
        for node_type, csv_name in NODE_TYPE_TO_CSV.items():
            csv_path = data_dir / csv_name
            if not csv_path.exists():
                # Try LGP alternate name
                if node_type == NodeType.LGP:
                    alt = data_dir / _LGP_ALT_CSV
                    if alt.exists():
                        csv_path = alt
                    else:
                        print(f"  [SKIP] No CSV for {node_type.name} in {data_dir}")
                        continue
                else:
                    print(f"  [SKIP] No CSV for {node_type.name} in {data_dir}")
                    continue
            try:
                result = evaluate_approximation_method(
                    csv_path=csv_path,
                    node_str=node_type.name,
                    quantile_levels=q_levels,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\nERROR in approximation evaluation for {node_type.name}: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Datasize experiment evaluation (--datasize_dir)
    # ------------------------------------------------------------------
    if datasize_dir is not None:
        print(f"\n{'='*60}")
        print(f"Running datasize experiment evaluation from: {datasize_dir}")
        print(f"Detail results will be saved to: {datasize_detail_output}")
        print(f"{'='*60}")
        agg_rows = evaluate_datasize_all(
            datasize_dir=datasize_dir,
            data_dir=data_dir,
            quantile_levels=q_levels,
            device=device,
            scripted=args.scripted,
            detail_output=datasize_detail_output,
        )
        all_results.extend(agg_rows)

    # ------------------------------------------------------------------
    # Save & print summary
    # ------------------------------------------------------------------
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(str(output), index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output}")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
