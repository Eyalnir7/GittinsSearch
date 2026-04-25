"""
Main script for training constraint GNN with Weights & Biases integration.
This script orchestrates dataset loading, model training, and experiment tracking.
"""
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import os
import json
import yaml
from functools import partial

# Import project modules
from bestScriptedModel import ScriptableConstraintGNN, forward_heteroBatch
from dataset import load_or_create_datasets
from training import train_model, evaluate_test, save_predictions_plot, evaluate_test_quantiles, save_quantile_predictions_plot
from pinBallLoss import pinball_loss_varlen
from enums import TaskType, NodeType

def get_minimum_achievable_loss(dataset, criterion):
    """Calculate minimum achievable loss for a dataset"""
    all_targets = []
    invalid_indices = []
    
    for i in range(len(dataset)):
        hetero_data = dataset[i]
        target_val = hetero_data.y.item()
        
        # Check for NaN or invalid values
        if target_val != target_val or target_val is None:  # NaN check
            invalid_indices.append(i)
            continue
        
        all_targets.append(target_val)
    
    if invalid_indices:
        print(f"WARNING: Found {len(invalid_indices)} samples with NaN/invalid targets")
        print(f"  Invalid indices: {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}")
        print("  Example invalid sample:")
        print(dataset[invalid_indices[0]])  # Print first invalid sample for debugging
    
    if not all_targets:
        print("ERROR: No valid targets found in dataset!")
        return float('nan'), torch.tensor([], dtype=torch.float)
    
    all_targets_tensor = torch.tensor(all_targets, dtype=torch.float)
    eps = 1e-7
    safe_targets = all_targets_tensor.clamp(eps, 1 - eps)
    optimal_logits = torch.log(safe_targets / (1 - safe_targets))
    min_loss = criterion(optimal_logits, safe_targets).item()
    
    return min_loss, all_targets_tensor


def _log_dataset_split_info(dataset_metadata, task, nodeType):
    """Log dataset split indices to WandB as an artifact"""

    train_indices = dataset_metadata.get("train_indices")
    val_indices = dataset_metadata.get("val_indices")
    test_indices = dataset_metadata.get("test_indices")

    if not (train_indices and val_indices and test_indices):
        return

    # ---- 1. Write splits to a local JSON file ----
    # Keep values JSON-serializable (enums are not serializable by default)
    splits = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "task": getattr(task, "name", task),
        "nodeType": getattr(nodeType, "name", nodeType),
        "source": dataset_metadata.get("source"),
    }
    
    # Name the split file after the current run for traceability
    splits_file = f"{wandb.run.name}.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f)

    # ---- 2. Create and log artifact ----
    log_dict = {
        "dataset/train_count": len(train_indices),
        "dataset/val_count": len(val_indices),
        "dataset/test_count": len(test_indices),
    }
    
    meta_data = {
            "train_count": len(train_indices),
            "val_count": len(val_indices),
            "test_count": len(test_indices),
        }

    if task == TaskType.FEASIBILITY:
        log_dict.update({
            "dataset/train_min_loss": dataset_metadata.get("train_min_loss"),
            "dataset/val_min_loss": dataset_metadata.get("val_min_loss"),
            "dataset/test_min_loss": dataset_metadata.get("test_min_loss"),
        })
        meta_data.update({
            "train_min_loss": dataset_metadata.get("train_min_loss"),
            "val_min_loss": dataset_metadata.get("val_min_loss"),
            "test_min_loss": dataset_metadata.get("test_min_loss"),
        })
    artifact_name = dataset_metadata.get("artifact_name", "dataset_splits")
    artifact = wandb.Artifact(
        name=f"{artifact_name}_splits",
        type="dataset-splits",
        description="Train/val/test indices used for this run",
        metadata=meta_data,
    )

    artifact.add_file(splits_file)
    logged_artifact = wandb.log_artifact(artifact)
    # ---- 3. Log lightweight info to metrics / summary ----
    wandb.log(log_dict)

    wandb.summary.update({
        "dataset/splits_artifact": logged_artifact.name,
        "dataset/train_count": len(train_indices),
        "dataset/val_count": len(val_indices),
        "dataset/test_count": len(test_indices),
    })

    # ---- 4. Track dataset provenance in config ----
    wandb.config.update(
        {
            "dataset/source": dataset_metadata.get("source"),
            "dataset/artifact_name": dataset_metadata.get("artifact_name"),
            "dataset/artifact_version": dataset_metadata.get("artifact_version"),
        },
        allow_val_change=True,
    )

    # Optional cleanup
    os.remove(splits_file)


def main(task, nodeType, use_wandb_datasets=True, datadir="../data/randomBlocks/", percentiles=[0.1, 0.3, 0.5, 0.7, 0.9], data_percentage=1.0, project="randomBlocks", seed=42, is_sweep=False, sweep_id=None, train_artifact_name=None, device=None):
    """Main training function with WANDB integration
    
    Args:
        use_wandb_datasets: If True, try to load datasets from WANDB artifacts
        task: Task type (e.g., "optimal_time_prediction_mse" or None for classification)
        seed: Random seed for reproducible dataset sampling
    """
    # Initialize WANDB with sweep support
    wandb.init(project=project)
    # Set custom run name with task, nodeType, seed, and wandb's unique run ID
    run_sweep_id = sweep_id or getattr(wandb.run, "sweep_id", None)
    percentiles_str = None
    if task in [TaskType.QUANTILE_REGRESSION_FEAS, TaskType.QUANTILE_REGRESSION_INFEAS]:
        percentiles_str = "_".join([f"{int(p*100)}" for p in percentiles])
        run_name = f"{task.name}_{nodeType.name}_s{seed}_p{percentiles_str}_{wandb.run.id}"
    else:
        run_name = f"{task.name}_{nodeType.name}_s{seed}_{wandb.run.id}"
    wandb.run.name = run_name
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    
    # Get config from sweep (or use defaults)
    config = wandb.config
    # Record dataset percentage in the run config
    # config.hidden_dim = 128
    config.dataset_percent = data_percentage
    config.seed = seed
    config.task_type = task.name
    config.node_type = nodeType.name
    config.sweep_id = run_sweep_id
    # Record dataset name (last component of datadir) to distinguish models when training on different datasets
    dataset_name = os.path.basename(os.path.normpath(datadir))
    config.dataset_name = dataset_name

    if device is not None:
        device_available = torch.device(device)
    else:
        device_available = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # device_available = torch.device("cpu")
    
    # Set default hyperparameters if not in config
    _set_default_config(config, task, nodeType)
    
    # Set activation function based on config
    if config.activation_function == "gelu":
        activation_function = nn.GELU()
    elif config.activation_function == "prelu":
        activation_function = nn.PReLU()
    else:
        activation_function = nn.ReLU()  # Default to ReLU if unknown
    
    # Fixed hyperparameters
    config.num_epochs = 10
    config.patience = 30  # Number of epochs to wait for improvement
    config.percentiles_tensor = torch.tensor(percentiles, dtype=torch.float)
    
    # Set working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    model_dir = "models"
    if is_sweep and run_sweep_id:
        model_dir = os.path.join(model_dir, "sweeps", run_sweep_id)
        os.makedirs(model_dir, exist_ok=True)

    # Set loss criterion and output transform based on task
    if task == TaskType.FEASIBILITY:
        criterion = nn.BCEWithLogitsLoss()
        output_transform = nn.Sigmoid()
    elif task in [TaskType.QUANTILE_REGRESSION_FEAS, TaskType.QUANTILE_REGRESSION_INFEAS]:
        taus = torch.tensor(percentiles, dtype=torch.float)
        criterion = partial(pinball_loss_varlen, taus=taus, reduction="mean")
        output_transform = lambda x: x  # Identity
    
    # Load or create datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset, dataset_metadata = load_or_create_datasets(
        scenes_dir=datadir,
        use_wandb_datasets=use_wandb_datasets,
        task=task,
        nodeType=nodeType,
        return_metadata=True,
        percentiles=percentiles_str,
        percent_data=data_percentage,
        seed=seed,
        train_artifact_name=train_artifact_name,
    )


    if task==TaskType.FEASIBILITY:
        print("\nCalculating minimum achievable losses...")
        train_min_loss, train_targets_tensor = get_minimum_achievable_loss(train_dataset, criterion)
        val_min_loss, val_targets_tensor = get_minimum_achievable_loss(val_dataset, criterion)
        test_min_loss, test_targets_tensor = get_minimum_achievable_loss(test_dataset, criterion)
        print(f"Train - Min loss: {train_min_loss:.6f}, Avg target: {train_targets_tensor.mean().item():.6f}")
        print(f"Val   - Min loss: {val_min_loss:.6f}, Avg target: {val_targets_tensor.mean().item():.6f}")
        print(f"Test  - Min loss: {test_min_loss:.6f}, Avg target: {test_targets_tensor.mean().item():.6f}")
        dataset_metadata["train_min_loss"] = train_min_loss
        dataset_metadata["val_min_loss"] = val_min_loss
        dataset_metadata["test_min_loss"] = test_min_loss
    
    # Create dataloaders (num_workers=0 to avoid multiprocessing issues on macOS)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    _log_dataset_split_info(dataset_metadata, task, nodeType)
    
    # Initialize model
    if task in [TaskType.QUANTILE_REGRESSION_FEAS, TaskType.QUANTILE_REGRESSION_INFEAS]:
        output_dim = len(percentiles)
    else:
        output_dim = 1  # Binary classification
    model = ScriptableConstraintGNN(output_dim=output_dim, 
                                    hidden_dim=config.hidden_dim,
                                    dropout_rate=config.dropout_rate,
                                    pe_dim=config.pe_dim,
                                    num_message_passing_layers=config.num_message_passing_layers,
                                    use_layer_norm=bool(config.use_layer_norm),
                                    activation=activation_function)
    model.to(device_available)
    print(f"model device: {device_available}")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Training loop with early stopping
    # Include dataset name in the model filename to allow separate models per dataset
    model_save_path = os.path.join(model_dir, f"best_constraint_gnn_model_{dataset_name}_{wandb.run.id}.pt")
    
    best_val_loss, best_epoch, early_stopped, total_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        output_transform=output_transform,
        config=config,
        device=device_available,
        forward_fn=forward_heteroBatch,
        model_save_path=model_save_path,
        taskType=task
    )
    
    # Test evaluation
    print("\n" + "="*60)
    print("Evaluating on test dataset...")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_available)
    
    if task in [TaskType.QUANTILE_REGRESSION_FEAS, TaskType.QUANTILE_REGRESSION_INFEAS]:
        avg_test_loss, test_pred, test_targ, test_ids = evaluate_test_quantiles(
            model, test_loader, criterion, torch.tensor(percentiles, dtype=torch.float), device_available, forward_heteroBatch
        )
        # Save test plot for quantiles
        test_plot_path = save_quantile_predictions_plot(test_targ, test_pred, test_ids, percentiles,
                                                        phase="test")
    else:
        avg_test_loss, test_pred, test_targ, test_ids = evaluate_test(
            model, test_loader, criterion, output_transform, device_available, forward_heteroBatch
        )
        # Save test plot
        test_plot_path = save_predictions_plot(test_targ, test_pred, test_ids, 
                                            phase="test", min_val=0, max_val=1, run_id=wandb.run.id)
    
    print(f"Test Loss: {avg_test_loss:.6f}")

    
    # Log final metrics
    wandb.log({
        "test/loss": avg_test_loss,
        "test/predictions": wandb.Image(test_plot_path),
        "final/test_loss": avg_test_loss,
        "final/best_val_loss": best_val_loss,
        "final/best_epoch": best_epoch,
        "final/early_stopped": early_stopped,
        "final/total_epochs": total_epochs,
    })
    
    # Print final summary
    _print_final_summary(early_stopped, total_epochs, config.num_epochs, best_epoch, 
                        avg_test_loss, best_val_loss, config.patience)
    
    # Upload model to wandb before deleting
    if os.path.exists(model_save_path):
        save_model = True
        best_meta_path = None
        if is_sweep:
            best_meta_path = os.path.join(model_dir, "best_model_meta.json")
            best_meta = _read_json_file(best_meta_path)
            if best_meta and best_val_loss >= best_meta.get("best_val_loss", float("inf")):
                save_model = False

        if save_model:
            model_meta = {
                "run_id": wandb.run.id,
                "run_name": wandb.run.name,
                "task": task.name,
                "node_type": nodeType.name,
                "dataset_name": dataset_name,
                "dataset_percent": data_percentage,
                "seed": seed,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "early_stopped": early_stopped,
                "total_epochs": total_epochs,
                "sweep_id": run_sweep_id,
                "is_sweep": is_sweep,
                "model_path": os.path.basename(model_save_path),
            }
            model_meta_path = os.path.join(model_dir, f"model_meta_{wandb.run.id}.json")
            _write_json_file(model_meta_path, model_meta)

            if is_sweep and best_meta_path:
                _delete_previous_best_artifacts(model_dir, best_meta)
                best_meta = {
                    "best_val_loss": best_val_loss,
                    "run_id": wandb.run.id,
                    "model_path": os.path.basename(model_save_path),
                    "meta_path": os.path.basename(model_meta_path),
                }
                _write_json_file(best_meta_path, best_meta)

            artifact = wandb.Artifact(
                f"best_model_{task.name}_{nodeType.name}_s{seed}_p{str(data_percentage)}_{dataset_name}",
                type="model",
                metadata={
                    "best_val_loss": best_val_loss,
                    "best_test_loss": avg_test_loss,
                    "best_epoch": best_epoch,
                    "sweep_id": run_sweep_id,
                    "is_sweep": is_sweep,
                    "run_config": config
                },
            )
            artifact.add_file(model_save_path)
            artifact.add_file(model_meta_path)
            wandb.log_artifact(artifact)
        os.remove(model_save_path)
    
    wandb.finish()


def _set_default_config(config, task, nodeType):
    """Set default hyperparameter values if not in config, based on task and nodeType."""
    # Lookup table: (task, nodeType) -> default hyperparameters
    _DEFAULTS = {
        (TaskType.QUANTILE_REGRESSION_FEAS, NodeType.LGP): dict(
            hidden_dim=256,
            learning_rate=0.00022969757920541503,
            weight_decay=0.00003960003330808008,
            batch_size=128,
            dropout_rate=0.10817224843866365,
            num_message_passing_layers=5,
            pe_dim=4,
            activation_function="prelu",
            use_layer_norm=False,
        ),
        (TaskType.FEASIBILITY, NodeType.WAYPOINTS): dict(
            hidden_dim=256,
            learning_rate=0.00002924299459484551,
            weight_decay=0.00000766845795828903,
            batch_size=8,
            dropout_rate=0.22153991223442393,
            num_message_passing_layers=5,
            pe_dim=8,
            activation_function="prelu",
            use_layer_norm=True,
        ),
        (TaskType.QUANTILE_REGRESSION_INFEAS, NodeType.WAYPOINTS): dict(
            hidden_dim=256,
            learning_rate=0.0011006902231555215,
            weight_decay=0.0008035780384858212,
            batch_size=128,
            dropout_rate=0.1243061629894016,
            num_message_passing_layers=7,
            pe_dim=16,
            activation_function="gelu",
            use_layer_norm=True,
        ),
        (TaskType.QUANTILE_REGRESSION_FEAS, NodeType.RRT): dict(
            hidden_dim=256,
            learning_rate=0.0017557598822713011,
            weight_decay=0.0000070271867149727,
            batch_size=128,
            dropout_rate=0.1719570500358632,
            num_message_passing_layers=3,
            pe_dim=4,
            activation_function="gelu",
            use_layer_norm=False,
        ),
        (TaskType.QUANTILE_REGRESSION_INFEAS, NodeType.LGP): dict(
            hidden_dim=256,
            learning_rate=0.007140010376059002,
            weight_decay=0.0012171125743514314,
            batch_size=128,
            dropout_rate=0.17595792484461237,
            num_message_passing_layers=5,
            pe_dim=16,
            activation_function="prelu",
            use_layer_norm=False,
        ),
        (TaskType.FEASIBILITY, NodeType.LGP): dict(
            hidden_dim=256,
            learning_rate=0.0017584569867658884,
            weight_decay=0.00000862756245465477,
            batch_size=128,
            dropout_rate=0.48104130617360463,
            num_message_passing_layers=3,
            pe_dim=16,
            activation_function="relu",
            use_layer_norm=True,
        ),
        (TaskType.QUANTILE_REGRESSION_FEAS, NodeType.WAYPOINTS): dict(
            hidden_dim=256,
            learning_rate=0.0003134702479712767,
            weight_decay=0.00002216014630930446,
            batch_size=128,
            dropout_rate=0.12907134147015792,
            num_message_passing_layers=5,
            pe_dim=16,
            activation_function="prelu",
            use_layer_norm=False,
        ),
    }

    defaults = _DEFAULTS.get((task, nodeType))
    if defaults is None:
        raise ValueError(f"No default config defined for task={task}, nodeType={nodeType}")

    for key, value in defaults.items():
        if key not in config:
            setattr(config, key, value)


def _read_json_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_json_file(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _delete_previous_best_artifacts(model_dir, best_meta):
    if not best_meta:
        return
    prev_model = best_meta.get("model_path")
    prev_meta = best_meta.get("meta_path")
    if prev_model:
        _safe_remove(os.path.join(model_dir, prev_model))
    if prev_meta:
        _safe_remove(os.path.join(model_dir, prev_meta))


def _safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


def _print_final_summary(early_stopped, total_epochs, max_epochs, best_epoch, 
                        avg_test_loss, best_val_loss, patience):
    """Print final training summary"""
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Training completed: {'Early stopped' if early_stopped else 'Full training'}")
    print(f"Total epochs run: {total_epochs}/{max_epochs}")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Final test loss: {avg_test_loss:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    if early_stopped:
        print(f"Early stopping patience: {patience} epochs")
    print("="*60)


def run_sweep(sweep_id=None, count=10, use_wandb_datasets=True, task=None, nodeType=None, datadir="../data/randomBlocks/", percentiles=[0.1, 0.3, 0.5, 0.7, 0.9], data_percentage=1.0, project="randomBlocks", seed=42, train_artifact_name=None, device=None):
    """Run a hyperparameter sweep
    
    Args:
        sweep_id: If provided, run as an agent on existing sweep. Otherwise create new sweep.
        count: Number of runs for the sweep
        use_wandb_datasets: Whether to use WANDB datasets or create new ones
        task: Task type (e.g., "optimal_time_prediction_mse" or None)
        nodeType: Node type for the dataset
        datadir: Directory containing the dataset
        percentiles: Array of percentiles for the model to predict
        seed: Random seed for reproducible dataset sampling
    """
    if sweep_id is None:
        # Load and create new sweep from config file
        config_path = os.path.join(os.path.dirname(__file__), "sweep_config.yaml")
        
        with open(config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        print(f"Loaded sweep config with parameters:")
        for param_name in sweep_config.get('parameters', {}).keys():
            print(f"  - {param_name}")
        
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"Created new sweep: {sweep_id}")
    
    # Create a partial function with use_wandb_datasets and datadir fixed
    main_with_wandb = partial(main, task=task, nodeType=nodeType, use_wandb_datasets=use_wandb_datasets, datadir=datadir, percentiles=percentiles, data_percentage=data_percentage, project=project, seed=seed, is_sweep=True, sweep_id=sweep_id, train_artifact_name=train_artifact_name, device=device)
    
    # Run sweep with specified number of trials
    wandb.agent(sweep_id, function=main_with_wandb, count=count, project=project)

def string_to_task(task_str):
    """Convert string to task type or None"""
    if task_str == "feasibility":
        return TaskType.FEASIBILITY
    elif task_str == "feas_quantile":
        return TaskType.QUANTILE_REGRESSION_FEAS
    elif task_str == "infeas_quantile":
        return TaskType.QUANTILE_REGRESSION_INFEAS
    else:
        return None

def string_to_node_type(node_type_str):
    """Convert string to node type or None"""
    if node_type_str == "waypoints":
        return NodeType.WAYPOINTS
    elif node_type_str == "rrt":
        return NodeType.RRT
    elif node_type_str == "lgp":
        return NodeType.LGP
    else:
        return None


if __name__ == "__main__":
    import argparse
    
    # set working dir to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the working directory to that directory
    os.chdir(current_dir)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train constraint GNN with Weights & Biases integration")
    parser.add_argument("--no-wandb", action="store_true", 
                        help="Disable WANDB dataset loading")
    parser.add_argument("--datadir", type=str, default="../data/randomBlocks_all_splits/",
                        help="Directory containing the dataset")
    parser.add_argument("--task", type=str, default=None,
                        help="Task type options: feasibility, feas_quantile, infeas_quantile")
    parser.add_argument("--node-type", type=str, default="all",
                        help="Node type options: waypoints, rrt, lgp")
    parser.add_argument("--percentiles", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="Percentiles for the model to predict (e.g., --percentiles 0.1 0.3 0.5 0.7 0.9)")
    parser.add_argument("--data-percentage", type=float, default=1.0,
                        help="Percentage of the dataset to use (between 0 and 1)")
    parser.add_argument("--project", type=str, default="randomBlocks",
                        help="WANDB project name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible dataset sampling (default: 42)")
    parser.add_argument("--train-artifact-name", type=str, default=None,
                        help="Override the train artifact name in WANDB (e.g. '2blocks3blocksTrain-train'). "
                             "If not set, defaults to '{artifact_base_name}-train'.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0', 'mps'). "
                             "If not set, auto-detects the best available device.")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Sweep subcommand
    sweep_parser = subparsers.add_parser("sweep", help="Run hyperparameter sweep")
    sweep_parser.add_argument("count", type=int, nargs="?", default=20,
                             help="Number of trials for the sweep")
    sweep_parser.add_argument("sweep_id", type=str, nargs="?", default=None,
                             help="Existing sweep ID to run as an agent")
    
    args = parser.parse_args()
    
    # Convert arguments to variables
    use_wandb_datasets = not args.no_wandb
    datadir = args.datadir
    task = string_to_task(args.task)
    if task is None:
        raise ValueError(f"Unknown task type: {args.task}")
    
    nodeType = string_to_node_type(args.node_type)
    if nodeType is None:
        raise ValueError(f"Unknown node type: {args.node_type}")
    
    percentiles = args.percentiles
    project = args.project
    seed = args.seed
    train_artifact_name = args.train_artifact_name
    
    # Validate data percentage
    if not (0.0 < args.data_percentage <= 1.0):
        raise ValueError(f"--data-percentage must be in (0, 1], got: {args.data_percentage}")

    if args.command == "sweep":
        # Run hyperparameter sweep
        print("sweep count:", args.count)
        print("sweep id:", args.sweep_id)
        print(f"use_wandb_datasets: {use_wandb_datasets}")
        print(f"Using data percentage: {args.data_percentage}")
        print(f"project: {project}")
        
        print(f"Starting hyperparameter sweep with {args.count} trials...")
        run_sweep(sweep_id=args.sweep_id, count=args.count, 
                 use_wandb_datasets=use_wandb_datasets, task=task, nodeType=nodeType, datadir=datadir, percentiles=percentiles, data_percentage=args.data_percentage, project=project, seed=seed, train_artifact_name=train_artifact_name, device=args.device)
    else:
        # Run single training
        print(f"Using data percentage: {args.data_percentage}")
        print(f"Using seed: {seed}")
        main(task=task, nodeType=nodeType, use_wandb_datasets=use_wandb_datasets, datadir=datadir, percentiles=percentiles, data_percentage=args.data_percentage, project=project, seed=seed, train_artifact_name=train_artifact_name, device=args.device)
