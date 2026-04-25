#!/usr/bin/env python3
"""
eval_dir.py

Evaluate all non-scripted models in a directory on a data directory and
produce actual-vs-predicted plots, reusing plot helpers from training.py.

Usage
-----
    python eval_dir.py \
        --models-dir non_scripted_models/ \
        --data-dir   /path/to/data/dir/ \
        [--output-dir eval_plots/] \
        [--device cpu] \
        [--percentiles 0.1 0.3 0.5 0.7 0.9] \
        [--batch-size 32]

Expected data-dir layout
-------------------------
    aggregated_configurations.json
    aggregated_waypoints.csv
    aggregated_rrt_by_action.csv
    aggregated_lgp_by_plan.csv

Expected model filename format
-------------------------------
    model_<TASK>_<NODETYPE>_s<seed>_p<percent>_...pt
    e.g.  model_FEASIBILITY_WAYPOINTS_s42_p1.0_randomBlocks_all.pt
          model_QUANTILE_REGRESSION_FEAS_LGP_s42_p1.0_randomBlocks_all.pt

Plots are written to <output-dir>/figures/ using the model stem as the
phase name so each file is uniquely named.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make Learning/ importable when running from any cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent   # test_compile/
_LEARNING_DIR = _SCRIPT_DIR.parent              # Learning/
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

from bestScriptedModel import ScriptableConstraintGNN, forward_heteroBatch   # noqa: E402
from dataset import HeteroGraphDataset                                        # noqa: E402
from enums import TaskType, NodeType                                          # noqa: E402
from functools import partial                                                 # noqa: E402
from training import (                                                        # noqa: E402
    save_predictions_plot,
    save_quantile_predictions_plot,
    split_batch_targets,
    evaluate_test,
    evaluate_test_quantiles,
)
from pinBallLoss import pinball_loss_varlen                                   # noqa: E402
from torch_geometric.loader import DataLoader                                 # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PERCENTILES = [0.1, 0.3, 0.5, 0.7, 0.9]

# Order matters: longer prefixes must come first so they match before a shorter one
_TASK_PATTERNS = [
    ("QUANTILE_REGRESSION_INFEAS", TaskType.QUANTILE_REGRESSION_INFEAS),
    ("QUANTILE_REGRESSION_FEAS",   TaskType.QUANTILE_REGRESSION_FEAS),
    ("FEASIBILITY",                TaskType.FEASIBILITY),
]
_NODE_PATTERNS = [
    ("WAYPOINTS", NodeType.WAYPOINTS),
    ("RRT",       NodeType.RRT),
    ("LGP",       NodeType.LGP),
]


# ---------------------------------------------------------------------------
# Parse task / node type from the model filename stem
# ---------------------------------------------------------------------------
def parse_task_and_node_type(model_stem: str):
    """Return (TaskType, NodeType) parsed from the model filename stem, or (None, None)."""
    upper = model_stem.upper()
    task_type = None
    for pattern, tt in _TASK_PATTERNS:
        if pattern in upper:
            task_type = tt
            break

    node_type = None
    for pattern, nt in _NODE_PATTERNS:
        if pattern in upper:
            node_type = nt
            break

    return task_type, node_type


# ---------------------------------------------------------------------------
# Load a non-scripted model from a checkpoint saved by script_model.py
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: Path, device: torch.device) -> ScriptableConstraintGNN:
    """Reconstruct ScriptableConstraintGNN from a run_config-bearing checkpoint."""
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    print(raw.keys())
    state_dict = raw["model_state_dict"]
    run_config  = raw["run_config"]

    hidden_dim          = run_config["hidden_dim"]
    pe_dim              = run_config["pe_dim"]
    dropout_rate        = run_config["dropout_rate"]
    num_layers          = run_config["num_message_passing_layers"]
    activation_function = run_config["activation_function"]

    # Always derive these two from the state dict — run_config may be stale
    use_layer_norm     = any(k.startswith("input_norms.") for k in state_dict)
    output_layer_w     = state_dict.get("output_layer.layer.5.weight")
    output_dim         = output_layer_w.shape[0] if output_layer_w is not None else run_config.get("output_dim", 1)

    print(f"    hidden_dim={hidden_dim}  pe_dim={pe_dim}  layers={num_layers}  "
          f"ln={use_layer_norm}  act={activation_function}  output_dim={output_dim}")

    if activation_function == "prelu":
        activation: nn.Module = nn.PReLU()
    elif activation_function == "gelu":
        activation = nn.GELU()
    else:
        activation = nn.ReLU()

    model = ScriptableConstraintGNN(
        hidden_dim=hidden_dim,
        pe_dim=pe_dim,
        num_message_passing_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=output_dim,
        use_layer_norm=use_layer_norm,
        activation=activation,
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _collect_file_ids(batch) -> list:
    if not hasattr(batch, "file_id"):
        return []
    fids = batch.file_id
    if isinstance(fids, (list, tuple)):
        return list(fids)
    if torch.is_tensor(fids):
        return fids.tolist()
    return [fids]


# ---------------------------------------------------------------------------
# Evaluate a single model
# ---------------------------------------------------------------------------
def evaluate_model(
    model_path: Path,
    data_dir: Path,
    device: torch.device,
    percentiles: list,
    batch_size: int,
) -> bool:
    model_stem = model_path.stem   # e.g. "model_FEASIBILITY_WAYPOINTS_s42_p1.0_..."
    task_type, node_type = parse_task_and_node_type(model_stem)

    if task_type is None or node_type is None:
        print(f"  ✗ Cannot parse task/node type from '{model_stem}', skipping.")
        return False

    print(f"\n{'='*60}")
    print(f"Model     : {model_path.name}")
    print(f"Task      : {task_type.name}  |  Node type: {node_type.name}")

    # --- load model ---
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False

    # --- build dataset (no train/val/test split needed — use everything) ---
    try:
        dataset = HeteroGraphDataset(
            input_path=str(data_dir) + "/",
            nodeType=node_type,
            taskType=task_type,
            device=device,
            seed=42
        )
    except Exception as e:
        print(f"  ✗ Failed to build dataset: {e}")
        return False

    if len(dataset) == 0:
        print("  ✗ Dataset is empty, skipping.")
        return False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # phase name = model stem without "model_" prefix so each plot file is unique
    phase_name = model_stem.removeprefix("model_")

    # --- inference + plot ---
    try:
        if task_type == TaskType.FEASIBILITY:
            criterion = nn.BCEWithLogitsLoss()
            avg_loss, preds, targets, file_ids = evaluate_test(
                model, loader, criterion, torch.sigmoid, device, forward_heteroBatch
            )
            plot_path = save_predictions_plot(
                targets, preds, file_ids,
                phase=phase_name,
                min_val=0.0,
                max_val=1.0,
            )
            print(f"  ✓ {len(preds)} samples  |  BCE loss: {avg_loss:.6f}  |  plot: {plot_path}")

        else:
            taus = torch.tensor(percentiles, dtype=torch.float)
            criterion = partial(pinball_loss_varlen, taus=taus, reduction="mean")
            avg_loss, pred_list, target_list, file_ids = evaluate_test_quantiles(
                model, loader, criterion, taus, device, forward_heteroBatch
            )
            plot_path = save_quantile_predictions_plot(
                target_list, pred_list, file_ids,
                percentiles=percentiles,
                phase=phase_name,
            )
            print(f"  ✓ {len(pred_list)} samples  |  Pinball loss: {avg_loss:.6f}  |  plot: {plot_path}")

    except Exception as e:
        print(f"  ✗ Inference/plot failed: {e}")
        import traceback; traceback.print_exc()
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GNN models in a directory and produce actual-vs-predicted plots."
    )
    parser.add_argument("--models-dir", required=True,
                        help="Directory containing model_*.pt non-scripted checkpoints.")
    parser.add_argument("--data-dir", required=True,
                        help="Data directory with aggregated_*.csv and aggregated_configurations.json.")
    parser.add_argument("--output-dir", default="eval_plots",
                        help="Output directory for plots (default: eval_plots/).")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (default: cpu).")
    parser.add_argument("--percentiles", type=float, nargs="+", default=DEFAULT_PERCENTILES,
                        help="Quantile levels for quantile-regression models "
                             "(default: 0.1 0.3 0.5 0.7 0.9).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default: 32).")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    data_dir   = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    device     = torch.device(args.device)

    for path, flag in [(models_dir, "--models-dir"), (data_dir, "--data-dir")]:
        if not path.is_dir():
            sys.exit(f"Error: {flag} '{path}' is not a directory.")

    # training.py saves plots to "figures/<phase>_predictions_*.png" relative to cwd
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    os.chdir(output_dir)

    model_files = sorted(models_dir.glob("model_*.pt"))
    if not model_files:
        sys.exit(f"No model_*.pt files found in '{models_dir}'.")

    print(f"Models dir : {models_dir}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {device}")
    print(f"Percentiles: {args.percentiles}")
    print(f"Found {len(model_files)} model(s).\n{'='*60}")

    passed, failed = 0, []
    for model_path in model_files:
        ok = evaluate_model(
            model_path=model_path,
            data_dir=data_dir,
            device=device,
            percentiles=args.percentiles,
            batch_size=args.batch_size,
        )
        if ok:
            passed += 1
        else:
            failed.append(model_path.name)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{len(model_files)} models evaluated successfully.")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  ✗ {name}")
    else:
        print(f"All plots written to: {output_dir}/figures/")


if __name__ == "__main__":
    main()
