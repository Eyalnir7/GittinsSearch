#!/usr/bin/env python3
"""
verify_scripted_models.py

Checks that each scripted TorchScript model (model_*.pt in --scripted-dir) produces
outputs numerically identical to the corresponding Python ScriptableConstraintGNN
model loaded from its state dict (model_*.pt in --non-scripted-dir).

For every matched pair the script:
  1. Builds a random but structurally valid HeteroGraph input.
  2. Runs a forward pass through both the Python model and the scripted model.
  3. Asserts that the outputs are close (within atol/rtol).

Usage
-----
    python verify_scripted_models.py \
        --scripted-dir    tuned_scripted_models/ \
        --non-scripted-dir non_scripted_models/ \
        [--atol 1e-5] [--rtol 1e-5] [--device cpu]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make sure the Learning package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent   # test_compile/
_LEARNING_DIR = _SCRIPT_DIR.parent              # Learning/
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

from bestScriptedModel import ScriptableConstraintGNN  # noqa: E402


def _load_python_model(
    state_dict_path: Path,
    device: torch.device,
) -> ScriptableConstraintGNN:
    """Reconstruct a ScriptableConstraintGNN from a checkpoint saved by script_model.py.

    All hyperparameters are read from the 'run_config' key stored in the checkpoint.
    """
    raw = torch.load(str(state_dict_path), map_location="cpu", weights_only=False)
    if "epoch" in raw:
        print(f"Found epoch in checkpoint: {raw['epoch']}")
    state_dict  = raw["model_state_dict"]
    run_config  = raw["run_config"]

    hidden_dim          = run_config["hidden_dim"]
    pe_dim              = run_config["pe_dim"]
    dropout_rate        = run_config["dropout_rate"]
    num_layers          = run_config["num_message_passing_layers"]
    activation_function = run_config["activation_function"]

    # Always derive use_layer_norm from the state dict — the run config may be stale
    # (model retrained with layer norm while config still says False).
    use_layer_norm = any(k.startswith("input_norms.") for k in state_dict)

    # Always derive output_dim from the state dict — it is often absent from run_config.
    output_layer_weight = state_dict.get("output_layer.layer.5.weight")
    if output_layer_weight is not None:
        output_dim = output_layer_weight.shape[0]
    else:
        output_dim = run_config.get("output_dim", 1)

    print(f"    [run_config] hidden_dim={hidden_dim}  pe_dim={pe_dim}  "
          f"layers={num_layers}  ln={use_layer_norm}  "
          f"act={activation_function}  output_dim={output_dim}")

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
# Random input generator
# ---------------------------------------------------------------------------

def _rand_inputs(device: torch.device, seed: int = 0):
    """
    Build a minimal but realistic set of dicts (x_dict, times_dict,
    edge_index_dict, batch_dict) as a single-graph batch.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    N = {
        "ssBox": 3,
        "place_frame": 3,
        "object": 3,
        "ssCylinder": 3,
        "pick": 4,
        "place": 4,
    }
    feat_dims = {"ssBox": 4, "place_frame": 4, "object": 4, "ssCylinder": 3}

    x_dict: dict[str, torch.Tensor] = {
        nt: torch.rand(n, feat_dims[nt], generator=rng, device=device)
        for nt, n in N.items()
        if nt in feat_dims
    }

    times_dict: dict[str, torch.Tensor] = {
        nt: torch.randint(0, 50, (n,), generator=rng, device=device)
        for nt, n in N.items()
    }

    batch_dict: dict[str, torch.Tensor] = {
        nt: torch.zeros(n, dtype=torch.long, device=device)
        for nt, n in N.items()
    }

    # A small number of edges per relation
    def _edges(src_n: int, dst_n: int, num_edges: int = 4) -> torch.Tensor:
        src = torch.randint(0, src_n, (num_edges,), generator=rng, device=device)
        dst = torch.randint(0, dst_n, (num_edges,), generator=rng, device=device)
        return torch.stack([src, dst], dim=0)

    edge_index_dict: dict[str, torch.Tensor] = {
        "object___close_edge___ssBox":          _edges(N["object"],      N["ssBox"]),
        "ssBox___close_edge___object":          _edges(N["ssBox"],       N["object"]),
        "place_frame___close_edge___ssBox":     _edges(N["place_frame"], N["ssBox"]),
        "ssBox___close_edge___place_frame":     _edges(N["ssBox"],       N["place_frame"]),
        "place_frame___close_edge___object":    _edges(N["place_frame"], N["object"]),
        "object___close_edge___place_frame":    _edges(N["object"],      N["place_frame"]),
        "pick___time_edge___place":             _edges(N["pick"],        N["place"]),
        "place___time_edge___pick":             _edges(N["place"],       N["pick"]),
        "object___time_edge___object":          _edges(N["object"],      N["object"]),
        "ssBox___time_edge___ssBox":            _edges(N["ssBox"],       N["ssBox"]),
        "place_frame___time_edge___place_frame": _edges(N["place_frame"], N["place_frame"]),
        "ssCylinder___time_edge___ssCylinder":  _edges(N["ssCylinder"],  N["ssCylinder"]),
        "object___pick_edge___pick":            _edges(N["object"],      N["pick"]),
        "pick___pick_edge___object":            _edges(N["pick"],        N["object"]),
        "place_frame___pick_edge___pick":       _edges(N["place_frame"], N["pick"]),
        "pick___pick_edge___place_frame":       _edges(N["pick"],        N["place_frame"]),
        "ssCylinder___pick_edge___pick":        _edges(N["ssCylinder"],  N["pick"]),
        "pick___pick_edge___ssCylinder":        _edges(N["pick"],        N["ssCylinder"]),
        "object___place_edge___place":          _edges(N["object"],      N["place"]),
        "place___place_edge___object":          _edges(N["place"],       N["object"]),
        "ssCylinder___place_edge___place":      _edges(N["ssCylinder"],  N["place"]),
        "place___place_edge___ssCylinder":      _edges(N["place"],       N["ssCylinder"]),
        "place_frame___place_edge___place":     _edges(N["place_frame"], N["place"]),
        "place___place_edge___place_frame":     _edges(N["place"],       N["place_frame"]),
    }

    return x_dict, times_dict, edge_index_dict, batch_dict


# ---------------------------------------------------------------------------
# Matching scripted <-> non-scripted pairs by filename stem
# ---------------------------------------------------------------------------

def _discover_pairs(
    scripted_dir: Path, non_scripted_dir: Path
) -> list[tuple[Path, Path]]:
    """
    Match model_*.pt files in both directories by their stem.
    Returns list of (scripted_path, non_scripted_path).
    """
    scripted = {p.name: p for p in sorted(scripted_dir.glob("model_*.pt"))}
    non_scripted = {p.name: p for p in sorted(non_scripted_dir.glob("model_*.pt"))}

    common = sorted(set(scripted) & set(non_scripted))
    only_scripted = sorted(set(scripted) - set(non_scripted))
    only_non_scripted = sorted(set(non_scripted) - set(scripted))

    if only_scripted:
        print(f"  [WARN] {len(only_scripted)} scripted model(s) have no non-scripted counterpart:")
        for n in only_scripted:
            print(f"         {n}")
    if only_non_scripted:
        print(f"  [WARN] {len(only_non_scripted)} non-scripted model(s) have no scripted counterpart:")
        for n in only_non_scripted:
            print(f"         {n}")

    return [(scripted[n], non_scripted[n]) for n in common]


# ---------------------------------------------------------------------------
# Single-pair verification
# ---------------------------------------------------------------------------

def verify_pair(
    scripted_path: Path,
    non_scripted_path: Path,
    device: torch.device,
    atol: float,
    rtol: float,
    num_seeds: int,
) -> bool:
    """
    Returns True if outputs match for all seeds, False otherwise.
    """
    print(f"\n  Scripted    : {scripted_path.name}")
    print(f"  Non-scripted: {non_scripted_path.name}")

    # Load scripted model
    try:
        scripted_model = torch.jit.load(str(scripted_path), map_location=device)
        scripted_model.eval()
    except Exception as e:
        print(f"  ✗ Failed to load scripted model: {e}")
        return False

    # Load Python model
    try:
        python_model = _load_python_model(non_scripted_path, device)
    except Exception as e:
        print(f"  ✗ Failed to load Python model: {e}")
        return False

    all_passed = True

    for seed in range(num_seeds):
        x_dict, times_dict, edge_index_dict, batch_dict = _rand_inputs(
            device, seed=seed
        )

        with torch.no_grad():
            try:
                out_scripted = scripted_model.forward(
                    x_dict, times_dict, edge_index_dict, batch_dict
                )
            except Exception as e:
                print(f"  ✗ Scripted forward failed (seed={seed}): {e}")
                all_passed = False
                continue

            try:
                out_python = python_model.forward(
                    x_dict, times_dict, edge_index_dict, batch_dict
                )
            except Exception as e:
                print(f"  ✗ Python forward failed (seed={seed}): {e}")
                all_passed = False
                continue

        passed = torch.allclose(out_scripted, out_python, atol=atol, rtol=rtol)
        max_diff = (out_scripted - out_python).abs().max().item()

        if passed:
            print(f"  ✓ seed={seed}  max_diff={max_diff:.2e}  outputs match")
        else:
            print(f"  ✗ seed={seed}  max_diff={max_diff:.2e}  MISMATCH")
            print(f"      scripted : {out_scripted.flatten().tolist()}")
            print(f"      python   : {out_python.flatten().tolist()}")
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify that scripted and Python models produce identical outputs."
    )
    parser.add_argument(
        "--scripted-dir",
        default="tuned_scripted",
        help="Directory containing TorchScript model_*.pt files.",
    )
    parser.add_argument(
        "--non-scripted-dir",
        default="tuned_non_scripted",
        help="Directory containing plain state-dict model_*.pt files.",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5,
        help="Absolute tolerance for torch.allclose (default: 1e-5).",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-5,
        help="Relative tolerance for torch.allclose (default: 1e-5).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of random inputs to test per model pair (default: 5).",
    )
    args = parser.parse_args()

    scripted_dir    = Path(args.scripted_dir).resolve()
    non_scripted_dir = Path(args.non_scripted_dir).resolve()
    device = torch.device(args.device)

    for d, label in [(scripted_dir, "--scripted-dir"), (non_scripted_dir, "--non-scripted-dir")]:
        if not d.is_dir():
            sys.exit(f"Error: {label} '{d}' is not a directory.")

    print(f"Scripted dir    : {scripted_dir}")
    print(f"Non-scripted dir: {non_scripted_dir}")
    print(f"Device          : {device}")
    print(f"atol={args.atol}  rtol={args.rtol}  seeds={args.num_seeds}")

    pairs = _discover_pairs(scripted_dir, non_scripted_dir)
    if not pairs:
        sys.exit("No matching model pairs found.")

    print(f"\nFound {len(pairs)} matching pair(s).\n{'='*60}")

    passed_count = 0
    failed_names: list[str] = []

    for scripted_path, non_scripted_path in pairs:
        ok = verify_pair(
            scripted_path, non_scripted_path,
            device=device,
            atol=args.atol,
            rtol=args.rtol,
            num_seeds=args.num_seeds,
        )
        if ok:
            passed_count += 1
        else:
            failed_names.append(scripted_path.name)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed_count}/{len(pairs)} pairs passed.")
    if failed_names:
        print("Failed models:")
        for n in failed_names:
            print(f"  ✗ {n}")
        sys.exit(1)
    else:
        print("All models match. ✓")


if __name__ == "__main__":
    main()
