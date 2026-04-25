#!/usr/bin/env python3
"""
Script to download all model artifacts from wandb and compile them using TorchScript.
"""

import torch
import torch.nn as nn
import wandb
import argparse
import os
import shutil
import re
import json
import traceback
# import from the parent folder
# add parent folder to sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bestScriptedModel import ScriptableConstraintGNN


def get_all_model_artifacts(project="randomBlocks", entity=None, require_p=True):
    """Get the latest version of all model artifacts from a wandb project
    
    If require_p is True (default), only artifacts whose base name contains a
    p{percent} token (e.g., p0.2) will be returned.
    
    Args:
        project: Wandb project name
        entity: Wandb entity (username or team), if None uses default
        require_p: If True, filter artifacts to only those containing p{percent}
        
    Returns:
        List of artifact names (latest version only)
    """
    print(f"\nFetching latest version of all model artifacts from project: {project}")
    
    api = wandb.Api()
    
    # Get the entity from the API if not provided
    if entity is None:
        entity = api.viewer.entity
        print(f"Using default entity: {entity}")
    
    project_path = f"{entity}/{project}"
    
    # Dictionary to store the latest version of each artifact
    # Key: artifact base name, Value: full_artifact_name with :latest alias
    latest_artifacts = {}
    
    # Regular expression to find p{percent} tokens
    p_pattern = re.compile(r'p\d+(?:\.\d+)?', re.IGNORECASE)
    
    # Get all model artifacts directly from the project
    try:
        # Use artifact_type to get all artifacts of type "model"
        artifact_type = api.artifact_type(type_name="model", project=project_path)
        
        # Get all artifact collections (base names without version)
        for collection in artifact_type.collections():
            # collection.name is like "best_model_FEASIBILITY_WAYPOINTS_p0.2_randomBlocks_all"
            base_name = collection.name
            if "all_splits" not in base_name:
                continue
            
            # If requested, only include artifacts that have a p{percent} token
            if require_p and not p_pattern.search(base_name):
                continue
            
            # Get the latest version by using the :latest alias
            latest_artifact_name = f"{project_path}/{base_name}:latest"
            latest_artifacts[base_name] = latest_artifact_name
    
    except Exception as e:
        print(f"Error fetching artifacts: {e}")
        return []
    
    # Extract artifact names sorted
    artifact_list = sorted(latest_artifacts.values())
    
    print(f"Found {len(artifact_list)} model artifacts (latest versions only):")
    for name in artifact_list:
        print(f"  - {name}")
    
    return artifact_list


def get_all_dataset_split_artifacts(project="randomBlocks", entity=None):
    """Get the latest version of all dataset-splits artifacts from a wandb project
    
    Args:
        project: Wandb project name
        entity: Wandb entity (username or team), if None uses default
        
    Returns:
        List of dataset-splits artifact names (latest version only)
    """
    print(f"\nFetching latest version of all dataset-splits artifacts from project: {project}")

    api = wandb.Api()

    if entity is None:
        entity = api.viewer.entity
        print(f"Using default entity: {entity}")

    project_path = f"{entity}/{project}"

    latest_artifacts = {}

    try:
        artifact_type = api.artifact_type(type_name="dataset-splits", project=project_path)

        for collection in artifact_type.collections():
            base_name = collection.name
            latest_artifact_name = f"{project_path}/{base_name}:latest"
            latest_artifacts[base_name] = latest_artifact_name

    except Exception as e:
        print(f"Error fetching dataset-splits artifacts: {e}")
        return []

    artifact_list = sorted(latest_artifacts.values())

    print(f"Found {len(artifact_list)} dataset-splits artifacts (latest versions only):")
    for name in artifact_list:
        print(f"  - {name}")

    return artifact_list


def download_model_from_wandb(artifact_name, project="randomBlocks"):
    """Download model artifact from wandb using API (no run creation)
    
    Args:
        artifact_name: Name of the artifact (e.g., "entity/project/best_model_FEASIBILITY_WAYPOINTS:v0")
        project: Wandb project name
        
    Returns:
        Tuple of (model_path, run_config) where run_config is the full wandb run config
        containing all hyperparameters (pe_dim, use_layer_norm, activation_function,
        hidden_dim, num_message_passing_layers, dropout_rate, etc.)
    """
    print(f"\nDownloading artifact: {artifact_name}")
    
    # Use API to download without creating a run
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type='model')
    artifact_dir = artifact.download()
    
    # Find the model file in the artifact directory
    model_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No .pt file found in artifact {artifact_name}")
    
    model_path = os.path.join(artifact_dir, model_files[0])
    print(f"  Downloaded to: {model_path}")
    
    # Get the full run config from the run that logged this artifact.
    # The run config contains all hyperparameters (pe_dim, use_layer_norm,
    # activation_function, hidden_dim, num_message_passing_layers, dropout_rate, ...).
    # Note: artifact.metadata only has sparse fields (best_val_loss, best_epoch, sweep_id).
    run = artifact.logged_by()
    run_config = dict(run.config) if run is not None else {}
    run_id = run.id if run is not None else None
    if not run_config:
        print("  Warning: could not retrieve run config; will use defaults")
    
    return model_path, run_config, run_id


def download_dataset_splits_from_wandb(artifact_name, output_dir="dataset_splits"):
    """Download dataset-splits artifact from wandb using API (no run creation)

    Args:
        artifact_name: Name of the artifact (e.g., "entity/project/dataset_splits_something:v0")
        output_dir: Directory to save downloaded splits file
        
    Returns:
        Path to the saved splits file, or None if not found
    """
    print(f"\nDownloading dataset-splits artifact: {artifact_name}")

    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="dataset-splits")
    artifact_dir = artifact.download()

    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(artifact_dir) if f.endswith('.json')]
    if not json_files:
        print(f"  ✗ No .json file found in dataset-splits artifact {artifact_name}")
        return None

    # Use the first json file found
    source_json = os.path.join(artifact_dir, json_files[0])

    artifact_parts = artifact_name.split('/')
    if len(artifact_parts) >= 3:
        base_name = artifact_parts[-1].split(':')[0]
    else:
        base_name = artifact_name.split(':')[0]

    dest_path = os.path.join(output_dir, f"{base_name}.json")
    shutil.copy(source_json, dest_path)
    print(f"  ✓ Dataset splits saved to: {dest_path}")

    return dest_path


def _parse_task_and_node_type(base_name):
    """Extract (task_name, node_type) from an artifact base name like
    'FEASIBILITY_WAYPOINTS_p0.2_randomBlocks_all' or
    'QUANTILE_REGRESSION_FEAS_LGP_p0.8_randomBlocks_all'.
    Returns (None, None) if parsing fails.
    """
    task_names = [
        "QUANTILE_REGRESSION_FEAS",
        "QUANTILE_REGRESSION_INFEAS",
        "FEASIBILITY",
    ]
    node_types = ["WAYPOINTS", "RRT", "LGP"]

    for task in task_names:
        if base_name.upper().startswith(task + "_"):
            remainder = base_name[len(task) + 1:]
            for node in node_types:
                if remainder.upper().startswith(node + "_") or remainder.upper() == node:
                    return task, node
    return None, None


def infer_output_dim_from_artifact_name(artifact_name):
    """Infer output dimension from artifact name
    
    Args:
        artifact_name: Name like "best_model_FEASIBILITY_WAYPOINTS:v0"
        
    Returns:
        Output dimension (1 for classification, 5 for quantile regression)
    """
    # Check if it's a quantile regression model
    if "QUANTILE_REGRESSION" in artifact_name.upper():
        return 5  # Default for quantile regression
    else:
        return 1  # Default for classification


def infer_num_layers_from_state_dict(state_dict):
    """Infer number of message passing layers from checkpoint state_dict
    
    Args:
        state_dict: The model state dict from checkpoint
        
    Returns:
        Number of message passing layers
    """
    max_layer_idx = -1
    for key in state_dict.keys():
        if key.startswith('hetero_conv_list.'):
            # Extract layer index from keys like "hetero_conv_list.3.convs..."
            parts = key.split('.')
            if len(parts) > 1:
                try:
                    layer_idx = int(parts[1])
                    max_layer_idx = max(max_layer_idx, layer_idx)
                except ValueError:
                    pass
    
    # Number of layers is max_layer_idx + 1
    if max_layer_idx >= 0:
        return max_layer_idx + 1
    else:
        return 5  # Default fallback


def script_single_model(artifact_name, project="randomBlocks", output_dir="scripted_models", non_scripted_dir="non_scripted_models", no_group=False):
    """Download, load, and script a single model
    
    Args:
        artifact_name: Wandb artifact name
        project: Wandb project name
        output_dir: Directory to save scripted models
        non_scripted_dir: Directory to save non-scripted models
        
    Returns:
        Tuple of (scripted_path, non_scripted_path) or (None, None) if failed
    """
    try:
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(non_scripted_dir, exist_ok=True)
        
        # Infer output dimension from artifact name
        output_dim = infer_output_dim_from_artifact_name(artifact_name)
        
        print(f"\n{'='*60}")
        print(f"Processing: {artifact_name}")
        print(f"{'='*60}")
        
        # Download model weights and the full run config (has all hyperparameters)
        model_path, run_config, run_id = download_model_from_wandb(artifact_name, project)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Read all hyperparameters from the run config, with sensible fallbacks
        hidden_dim = run_config.get('hidden_dim')
        pe_dim = run_config.get('pe_dim')
        dropout_rate = run_config.get('dropout_rate')
        num_message_passing_layers = run_config.get('num_message_passing_layers')
        activation_function = run_config.get('activation_function')
        # use_layer_norm: always infer from state_dict — the run config may be stale
        # (model retrained with layer norm but config still says False).
        use_layer_norm = any(k.startswith('input_norms.') for k in state_dict.keys())
        run_config['use_layer_norm'] = use_layer_norm

        # output_dim: always infer from state_dict — often absent from run_config.
        output_layer_weight = state_dict.get('output_layer.layer.5.weight')
        if output_layer_weight is not None:
            output_dim = output_layer_weight.shape[0]
        run_config['output_dim'] = output_dim
        
        if activation_function == 'prelu':
            activation = nn.PReLU()
        elif activation_function == 'gelu':
            activation = nn.GELU()
        else:
            activation = nn.ReLU()
        
        print(f"  Model config (from run config):")
        print(f"    hidden_dim: {hidden_dim}")
        print(f"    pe_dim: {pe_dim}")
        print(f"    num_message_passing_layers: {num_message_passing_layers}")
        print(f"    dropout_rate: {dropout_rate}")
        print(f"    use_layer_norm: {use_layer_norm}")
        print(f"    activation: {activation_function}")
        
        # Initialize model with correct config
        model = ScriptableConstraintGNN(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            pe_dim=pe_dim,
            num_message_passing_layers=num_message_passing_layers,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            activation=activation,
        )
        
        # Load state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"  Val loss: {checkpoint['val_loss']:.6f}")
        else:
            model.load_state_dict(checkpoint)
        
        # Set to eval mode
        model.eval()
        
        # Generate output filename
        # Remove version tag and extract just the model name (strip entity/project prefix)
        # qualified_name format: "entity/project/artifact_name:version"
        artifact_parts = artifact_name.split('/')
        if len(artifact_parts) >= 3:
            # Has entity/project/name format
            base_name = artifact_parts[-1].split(':')[0].replace('best_model_', '')
        else:
            # Simpler format
            base_name = artifact_name.split(':')[0].replace('best_model_', '')
        
        if no_group:
            model_output_dir = output_dir
            model_non_scripted_dir = non_scripted_dir
        else:
            # Attempt to extract p{percent} token and group by it
            p_match = re.search(r'(p\d+(?:\.\d+)?)', base_name, re.IGNORECASE)
            if p_match:
                p_token = p_match.group(1).lower()
            else:
                p_token = "unknown_p"

            # Extract seed from run config
            seed = run_config.get('seed', 'unknown')

            # Create subdirectories: datasize_{p}/seed_{seed}
            model_output_dir = os.path.join(output_dir, f"datasize_{p_token}", f"seed_{seed}")
            model_non_scripted_dir = os.path.join(non_scripted_dir, f"datasize_{p_token}", f"seed_{seed}")
            os.makedirs(model_output_dir, exist_ok=True)
            os.makedirs(model_non_scripted_dir, exist_ok=True)
        
        # Save non-scripted model
        non_scripted_path = os.path.join(model_non_scripted_dir, f"model_{base_name}.pt")
        if isinstance(checkpoint, dict):
            new_checkpoint = checkpoint.copy()
        else:
            # Original checkpoint was a bare state_dict
            new_checkpoint = {}

        # Replace / insert model weights from the *new* model instance
        new_checkpoint["model_state_dict"] = model.state_dict()

        # Add run config (full hyperparameters)
        new_checkpoint["run_config"] = dict(run_config)

        # Optional but very useful provenance
        new_checkpoint["wandb_run_id"] = run_id
        new_checkpoint["artifact_name"] = artifact_name
        torch.save(new_checkpoint, non_scripted_path)
        print(f"  ✓ Non-scripted model saved to: {non_scripted_path}")
        
        # Script the model
        scripted_model = torch.jit.script(model)
        
        output_path = os.path.join(model_output_dir, f"model_{base_name}.pt")
        
        # Save scripted model
        torch.jit.save(scripted_model, output_path)
        print(f"  ✓ Scripted model saved to: {output_path}")
        
        # Verify by loading
        loaded_model = torch.jit.load(output_path)
        print(f"  ✓ Verified: Scripted model loads successfully")

        return output_path, non_scripted_path
        
    except Exception as e:
        print(f"  ✗ Error processing {artifact_name}: {e}")
        traceback.print_exc()
        return None, None


def main(project="randomBlocks", entity=None, output_dir="scripted_models", non_scripted_dir="non_scripted_models", dataset_splits_dir="dataset_splits", no_group=False, artifact=None):
    """Main function to download and script all models from wandb
    
    Args:
        project: Wandb project name
        entity: Wandb entity (username or team)
        output_dir: Directory to save scripted models
        non_scripted_dir: Directory to save non-scripted models
        dataset_splits_dir: Directory to save dataset-splits artifacts
        no_group: If True, save all models directly into output_dir without p-token subdirectories
        artifact: If provided, only process this specific artifact name instead of fetching all
    """
    print(f"\n{'='*60}")
    print("Model Scripting Pipeline")
    print(f"{'='*60}")
    print(f"Project: {project}")
    print(f"Scripted models directory: {output_dir}")
    print(f"Non-scripted models directory: {non_scripted_dir}")
    print(f"Dataset splits directory: {dataset_splits_dir}")
    
    # Get all model artifacts (or use the single specified artifact)
    if artifact is not None:
        api = wandb.Api()
        if entity is None:
            entity = api.viewer.entity
        # Normalise: if artifact already contains '/' treat as fully qualified,
        # otherwise prepend entity/project
        if '/' in artifact:
            artifact_names = [artifact]
        else:
            artifact_names = [f"{entity}/{project}/{artifact}"]
        print(f"\nUsing specified artifact: {artifact_names[0]}")
    else:
        artifact_names = get_all_model_artifacts(project, entity)
    
    # Process models
    model_successful = []
    model_failed = []
    
    if not artifact_names:
        print("\nNo model artifacts found in the project!")
    else:
        for i, artifact_name in enumerate(artifact_names, 1):
            print(f"\n[{i}/{len(artifact_names)}]")
            scripted_path, non_scripted_path = script_single_model(artifact_name, project, output_dir, non_scripted_dir, no_group=no_group)
            
            if scripted_path and non_scripted_path:
                model_successful.append((artifact_name, scripted_path, non_scripted_path))
            else:
                model_failed.append(artifact_name)

    # Get all dataset-splits artifacts (skip when a specific model artifact was requested)
    if artifact is not None:
        splits_artifacts = []
    else:
        splits_artifacts = get_all_dataset_split_artifacts(project, entity)
    splits_successful = []
    splits_failed = []

    if not splits_artifacts:
        print("\nNo dataset-splits artifacts found in the project!")
    else:
        for i, artifact_name in enumerate(splits_artifacts, 1):
            print(f"\n[Dataset splits {i}/{len(splits_artifacts)}]")
            splits_path = download_dataset_splits_from_wandb(artifact_name, dataset_splits_dir)
            if splits_path:
                splits_successful.append((artifact_name, splits_path))
            else:
                splits_failed.append(artifact_name)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total model artifacts: {len(artifact_names)}")
    print(f"Models successfully scripted: {len(model_successful)}")
    print(f"Models failed: {len(model_failed)}")
    print(f"Total dataset-splits artifacts: {len(splits_artifacts)}")
    print(f"Dataset-splits successfully downloaded: {len(splits_successful)}")
    print(f"Dataset-splits failed: {len(splits_failed)}")
    
    if model_successful:
        print(f"\n✓ Successfully processed models:")
        for artifact_name, scripted_path, non_scripted_path in model_successful:
            print(f"  - {artifact_name}")
            print(f"    → Scripted: {scripted_path}")
            print(f"    → Non-scripted: {non_scripted_path}")
    
    if model_failed:
        print(f"\n✗ Failed to process models:")
        for artifact_name in model_failed:
            print(f"  - {artifact_name}")

    if splits_successful:
        print(f"\n✓ Successfully downloaded dataset-splits:")
        for artifact_name, splits_path in splits_successful:
            print(f"  - {artifact_name}")
            print(f"    → Splits file: {splits_path}")

    if splits_failed:
        print(f"\n✗ Failed to download dataset-splits:")
        for artifact_name in splits_failed:
            print(f"  - {artifact_name}")
    
    if no_group:
        print(f"\nAll scripted models saved directly to: {output_dir}/")
        print(f"All non-scripted models saved directly to: {non_scripted_dir}/")
    else:
        print(f"\nAll scripted models saved under subdirectories of: {output_dir}/ (grouped as datasize_{{p}}/seed_{{i}}/)")
        print(f"All non-scripted models saved under subdirectories of: {non_scripted_dir}/ (grouped as datasize_{{p}}/seed_{{i}}/)")

    print(f"All dataset-splits saved to: {dataset_splits_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and script all constraint GNN models from wandb")
    parser.add_argument("--project", type=str, default="randomBlocks",
                       help="Wandb project name")
    parser.add_argument("--entity", type=str, default=None,
                       help="Wandb entity (username or team)")
    parser.add_argument("--output-dir", type=str, default="scripted_models",
                       help="Directory to save scripted models")
    parser.add_argument("--non-scripted-dir", type=str, default="non_scripted_models",
                       help="Directory to save non-scripted models")
    parser.add_argument("--dataset-splits-dir", type=str, default="dataset_splits",
                       help="Directory to save dataset-splits artifacts")
    parser.add_argument("--no-group", action="store_true", default=False,
                       help="Save all models directly into output-dir without grouping by p token")
    parser.add_argument("--artifact", type=str, default=None,
                       help="Process only this specific artifact name (e.g. 'best_model_FEASIBILITY_WAYPOINTS_p0.2_randomBlocks_all:latest') instead of downloading all artifacts. Can be a bare name, entity/project/name:version, or any wandb artifact path.")
    
    args = parser.parse_args()
    
    main(project=args.project, entity=args.entity, output_dir=args.output_dir, non_scripted_dir=args.non_scripted_dir, dataset_splits_dir=args.dataset_splits_dir, no_group=args.no_group, artifact=args.artifact)
