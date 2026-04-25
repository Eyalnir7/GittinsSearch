#!/usr/bin/env python3
"""
Script to load already-downloaded model checkpoints and compile them using TorchScript.
Looks for .pt files in the artifacts directory and scripts them.
"""

import torch
import argparse
import os
import glob
# import from the parent folder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bestScriptedModel_no_norm import ScriptableConstraintGNN


def infer_output_dim_from_filename(filename):
    """Infer output dimension from filename
    
    Args:
        filename: Name like "best_constraint_gnn_model_xyz.pt"
        
    Returns:
        Output dimension (1 for classification, 5 for quantile regression)
    """
    # Check if it's a quantile regression model
    if "QUANTILE_REGRESSION" in filename.upper():
        return 5
    else:
        return 1


def extract_model_name_from_path(model_path):
    """Extract a clean model name from the model file path
    
    Args:
        model_path: Path like "seed_1/best_model_FEASIBILITY_WAYPOINTS_abc123.pt"
        
    Returns:
        Clean name like "FEASIBILITY_WAYPOINTS_abc123"
    """
    stem = os.path.splitext(os.path.basename(model_path))[0]  # e.g. "best_model_FEASIBILITY_WAYPOINTS_abc123"
    return stem.replace('best_model_', '', 1)


def script_model_from_checkpoint(checkpoint_path, output_dir="models"):
    """Load a checkpoint and script it
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_dir: Directory to save scripted models
        
    Returns:
        Path to scripted model or None if failed
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract model name from path
        model_name = extract_model_name_from_path(checkpoint_path)
        
        # Infer output dimension
        output_dim = infer_output_dim_from_filename(checkpoint_path)
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(checkpoint_path)}")
        print(f"Model name: {model_name}")
        print(f"Output dim: {output_dim}")
        print(f"{'='*60}")
        
        # Model hyperparameters
        hidden_dim = 128
        pe_dim = 4
        num_message_passing_layers = 3
        dropout_rate = 0.1420325863085976
        
        # Initialize model
        model = ScriptableConstraintGNN(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            pe_dim=pe_dim,
            num_message_passing_layers=num_message_passing_layers,
            dropout_rate=dropout_rate
        )
        
        # Load checkpoint
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            result = model.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                print(f"  [INFO] Ignored unexpected keys: {result.unexpected_keys}")
            if result.missing_keys:
                print(f"  [WARN] Missing keys (not loaded): {result.missing_keys}")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"  Val loss: {checkpoint['val_loss']:.6f}")
        else:
            result = model.load_state_dict(checkpoint, strict=False)
            if result.unexpected_keys:
                print(f"  [INFO] Ignored unexpected keys: {result.unexpected_keys}")
            if result.missing_keys:
                print(f"  [WARN] Missing keys (not loaded): {result.missing_keys}")
        
        # Set to eval mode and script
        model.eval()
        print("Scripting model...")
        scripted_model = torch.jit.script(model)
        
        # Save
        output_path = os.path.join(output_dir, f"model_{model_name}.pt")
        torch.jit.save(scripted_model, output_path)
        print(f"  ✓ Scripted model saved to: {output_path}")
        
        # Verify
        loaded_model = torch.jit.load(output_path)
        print(f"  ✓ Verified: Model loads successfully")
        
        return output_path
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(artifacts_dir="artifacts", output_dir="models"):
    """Main function to script all models from artifacts directory
    
    Args:
        artifacts_dir: Directory containing downloaded artifacts
        output_dir: Directory to save scripted models
    """
    print(f"\n{'='*60}")
    print("Model Scripting Pipeline (Local Artifacts)")
    print(f"{'='*60}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all .pt files in artifacts directory
    pattern = os.path.join(artifacts_dir, "**", "*.pt")
    checkpoint_files = glob.glob(pattern, recursive=True)
    
    if not checkpoint_files:
        print(f"\nNo .pt files found in {artifacts_dir}")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  - {f}")
    
    # Process each checkpoint
    successful = []
    failed = []
    
    for i, checkpoint_path in enumerate(checkpoint_files, 1):
        print(f"\n[{i}/{len(checkpoint_files)}]")
        output_path = script_model_from_checkpoint(checkpoint_path, output_dir)
        
        if output_path:
            successful.append((checkpoint_path, output_path))
        else:
            failed.append(checkpoint_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total checkpoints: {len(checkpoint_files)}")
    print(f"Successfully scripted: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully scripted models:")
        for checkpoint_path, output_path in successful:
            model_name = extract_model_name_from_path(checkpoint_path)
            print(f"  - {model_name}")
            print(f"    → {output_path}")
    
    if failed:
        print(f"\n✗ Failed to script:")
        for checkpoint_path in failed:
            print(f"  - {checkpoint_path}")
    
    print(f"\nAll scripted models saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script downloaded constraint GNN models from local artifacts")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts",
                       help="Directory containing downloaded artifacts")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save scripted models")
    
    args = parser.parse_args()
    
    main(artifacts_dir=args.artifacts_dir, output_dir=args.output_dir)
