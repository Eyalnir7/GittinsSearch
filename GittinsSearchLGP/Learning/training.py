"""
Training, validation, and testing loops for constraint GNN.
"""
import torch
import wandb
import matplotlib
matplotlib.use('Agg')   # <- ensure non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import os
from enums import TaskType


def split_batch_targets(batch):
    """
    Turn concatenated batch.y (shape: sum_i M_i) and batch.y_len (shape: B)
    into a Python list of length B, where element i is a tensor of shape (M_i,).
    """
    lengths = batch.y_len.view(-1)  # (B,)
    cum_lengths = torch.cumsum(lengths, dim=0)  # (B,)
    starts = torch.cat(
        [lengths.new_zeros(1), cum_lengths[:-1]]
    )  # (B,) starting indices
    ends = cum_lengths  # (B,) ending indices (exclusive)

    y_list = [batch.y[s:e] for s, e in zip(starts, ends)]
    return y_list


def save_predictions_plot(targets, predictions, file_ids, run_id, epoch=None, phase="train", min_val=0, max_val=0.25):
    """Save a scatter plot of predictions vs targets, coloring by directory prefix (before '__').

    If multiple prefixes are present, use distinct discrete colors and add a legend.
    If only one prefix is present, plot a single color without a legend.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize inputs
    str_ids = [str(fid) for fid in file_ids]
    prefixes = [s.split('__', 1)[0] if '__' in s else '' for s in str_ids]
    unique_prefixes = sorted(list(dict.fromkeys(prefixes)))  # preserve order

    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Label', fontsize=12)
    ax.set_ylabel('Predicted Label', fontsize=12)
    if epoch is not None:
        ax.set_title(f'{phase.capitalize()} Predictions (Epoch {epoch})', fontsize=14)
    else:
        ax.set_title(f'{phase.capitalize()} Set Predictions', fontsize=14)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)

    if len(unique_prefixes) <= 1:
        # Single group: plot all points in default color
        ax.scatter(targets, predictions, color='C0', alpha=0.7, s=50)
    else:
        # Multiple groups: plot each prefix separately and add legend
        cmap = plt.get_cmap('tab10')
        color_map = {pref: cmap(i % cmap.N) for i, pref in enumerate(unique_prefixes)}
        for pref in unique_prefixes:
            idxs = [i for i, p in enumerate(prefixes) if p == pref]
            if not idxs:
                continue
            t_vals = [targets[i] for i in idxs]
            p_vals = [predictions[i] for i in idxs]
            label = pref if pref != '' else '<no_prefix>'
            ax.scatter(t_vals, p_vals, color=color_map[pref], alpha=0.8, s=50, label=label)

        # Use legend to show directory mapping
        ax.legend(title='Directory', loc='best')

    # Save and return as image
    plt.tight_layout()
    if epoch is not None:
        save_path = f"figures/{phase}_predictions_epoch_{epoch}_{run_id}.png"
    else:
        save_path = f"figures/{phase}_predictions_final_{run_id}.png"
    plt.savefig(save_path, dpi=100)
    plt.close()

    return save_path


def save_quantile_predictions_plot(targets_list, predictions_list, file_ids, percentiles, epoch=None, phase="train"):
    """Save scatter plots of predicted quantiles vs true quantiles
    
    Args:
        targets_list: List of target arrays (variable length per sample)
        predictions_list: List of quantile prediction arrays (one per sample), shape (num_quantiles,)
        file_ids: List of file identifiers
        percentiles: List/array of percentile levels (e.g., [0.1, 0.5, 0.9])
        epoch: Optional epoch number
        phase: Phase name ("train", "val", "test")
    
    Returns:
        Path to the saved figure
    """
    import numpy as np
    
    num_quantiles = len(percentiles)
    
    # Compute true quantiles from targets for each sample
    true_quantiles_per_sample = []
    for targets in targets_list:
        if len(targets) == 0:
            true_quantiles_per_sample.append([np.nan] * num_quantiles)
        else:
            true_q = np.quantile(targets, percentiles, method="lower")
            true_quantiles_per_sample.append(true_q.tolist())
    
    # Create subplots for each quantile
    n_cols = min(3, num_quantiles)
    n_rows = (num_quantiles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    
    # Flatten axes for easier indexing
    if num_quantiles == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for q_idx in range(num_quantiles):
        ax = axes[q_idx]
        
        # Extract true and predicted values for this quantile
        true_vals = [true_quantiles_per_sample[i][q_idx] for i in range(len(predictions_list))]
        pred_vals = [predictions_list[i][q_idx] for i in range(len(predictions_list))]
        
        # Filter out NaN values
        valid_indices = [i for i in range(len(true_vals)) if not np.isnan(true_vals[i])]
        true_vals_clean = [true_vals[i] for i in valid_indices]
        pred_vals_clean = [pred_vals[i] for i in valid_indices]
        file_ids_clean = [file_ids[i] for i in valid_indices]
        
        if len(true_vals_clean) > 0:
                # Determine prefixes for coloring
                str_ids = [str(fid) for fid in file_ids_clean]
                prefixes = [s.split('__', 1)[0] if '__' in s else '' for s in str_ids]
                unique_prefixes = sorted(list(dict.fromkeys(prefixes)))

                # Perfect prediction line
                min_val = min(min(true_vals_clean), min(pred_vals_clean))
                max_val = max(max(true_vals_clean), max(pred_vals_clean))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

                ax.set_xlabel(f'True {percentiles[q_idx]:.0%} Quantile', fontsize=12)
                ax.set_ylabel(f'Predicted {percentiles[q_idx]:.0%} Quantile', fontsize=12)
                ax.set_title(f'{percentiles[q_idx]:.0%} Quantile Predictions', fontsize=14)
                ax.grid(True, alpha=0.3)

                if len(unique_prefixes) <= 1:
                    # Single group: plot all points in a single color
                    ax.scatter(true_vals_clean, pred_vals_clean, color='C0', alpha=0.7, s=50)
                else:
                    cmap = plt.get_cmap('tab10')
                    color_map = {pref: cmap(i % cmap.N) for i, pref in enumerate(unique_prefixes)}
                    for pref in unique_prefixes:
                        idxs = [i for i, p in enumerate(prefixes) if p == pref]
                        if not idxs:
                            continue
                        t_vals = [true_vals_clean[i] for i in idxs]
                        p_vals = [pred_vals_clean[i] for i in idxs]
                        label = pref if pref != '' else '<no_prefix>'
                        ax.scatter(t_vals, p_vals, color=color_map[pref], alpha=0.8, s=50, label=label)

                    # Add legend for directories
                    ax.legend(title='Directory', loc='best')
    # Hide unused subplots
    for q_idx in range(num_quantiles, len(axes)):
        axes[q_idx].axis('off')
    
    # Set overall title
    if epoch is not None:
        fig.suptitle(f'{phase.capitalize()} Quantile Predictions (Epoch {epoch})', fontsize=16, y=0.995)
    else:
        fig.suptitle(f'{phase.capitalize()} Set Quantile Predictions', fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    if epoch is not None:
        save_path = f"figures/{phase}_quantile_predictions_epoch_{epoch}.png"
    else:
        save_path = f"figures/{phase}_quantile_predictions_final.png"
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def train_epoch(model, train_loader, optimizer, criterion, output_transform, epoch, device, forward_fn):
    """Train for one epoch and return predictions
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        output_transform: Transform to apply to output (e.g., Sigmoid)
        epoch: Current epoch number
        device: Device to run on
        forward_fn: Function to forward pass heterogeneous batch through model
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    print(f"model device: {next(model.parameters()).device}")
    train_loss = 0.0
    train_predictions = []
    train_targets = []
    train_file_ids = []
    
    for graphs in train_loader:
        graphs = graphs.to(device)
        labels = graphs.y.to(device)
        optimizer.zero_grad()
        
        output = forward_fn(model, graphs)
        loss = criterion(output, labels)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            train_predictions.extend(output_transform(output).detach().cpu().tolist())
            train_targets.extend(labels.detach().cpu().tolist())
            # Handle batched file_ids from PyTorch Geometric DataLoader
            if hasattr(graphs, 'file_id'):
                if isinstance(graphs.file_id, (list, tuple)):
                    train_file_ids.extend(graphs.file_id)
                elif torch.is_tensor(graphs.file_id):
                    train_file_ids.extend(graphs.file_id.tolist())
                else:
                    # Single file_id
                    train_file_ids.append(graphs.file_id)

    if epoch % 10 == 0:
        train_plot_path = save_predictions_plot(train_targets, train_predictions, train_file_ids, 
                                                    epoch=epoch, phase="train", 
                                                    min_val=0, max_val=1)
        wandb.log({f"train/predictions_epoch_{epoch}": wandb.Image(train_plot_path)})
    
    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


def validate_epoch(model, val_loader, criterion, output_transform, epoch, device, forward_fn):
    """Validate for one epoch and return predictions
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        output_transform: Transform to apply to output (e.g., Sigmoid)
        epoch: Current epoch number
        device: Device to run on
        forward_fn: Function to forward pass heterogeneous batch through model
    
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []
    val_file_ids = []
    
    with torch.no_grad():
        for graphs in val_loader:
            graphs = graphs.to(device)
            labels = graphs.y.to(device)
            output = forward_fn(model, graphs)
            loss = criterion(output, labels)
            val_loss += loss.item()

            if epoch % 10 == 0:
                val_predictions.extend(output_transform(output).detach().cpu().tolist())
                val_targets.extend(labels.detach().cpu().tolist())
                # Handle batched file_ids from PyTorch Geometric DataLoader
                if hasattr(graphs, 'file_id'):
                    if isinstance(graphs.file_id, (list, tuple)):
                        val_file_ids.extend(graphs.file_id)
                    elif torch.is_tensor(graphs.file_id):
                        val_file_ids.extend(graphs.file_id.tolist())
                    else:
                        # Single file_id
                        val_file_ids.append(graphs.file_id)
        
        if epoch % 10 == 0:
            val_plot_path = save_predictions_plot(val_targets, val_predictions, val_file_ids, 
                                                  epoch=epoch, phase="val", 
                                                  min_val=0, max_val=1)
            wandb.log({f"val/predictions_epoch_{epoch}": wandb.Image(val_plot_path)})
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def evaluate_test(model, test_loader, criterion, output_transform, device, forward_fn):
    """Evaluate on test set
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        output_transform: Transform to apply to output (e.g., Sigmoid)
        device: Device to run on
        forward_fn: Function to forward pass heterogeneous batch through model
    
    Returns:
        Tuple of (avg_test_loss, test_predictions, test_targets, test_file_ids)
    """
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    test_file_ids = []
    
    with torch.no_grad():
        for graphs in test_loader:
            graphs = graphs.to(device)
            labels = graphs.y.to(device)
            output = forward_fn(model, graphs)
            loss = criterion(output, labels)
            test_loss += loss.item()
            
            test_predictions.extend(output_transform(output).detach().cpu().tolist())
            test_targets.extend(labels.detach().cpu().tolist())
            # Handle batched file_ids from PyTorch Geometric DataLoader
            if hasattr(graphs, 'file_id'):
                if isinstance(graphs.file_id, (list, tuple)):
                    test_file_ids.extend(graphs.file_id)
                elif torch.is_tensor(graphs.file_id):
                    test_file_ids.extend(graphs.file_id.tolist())
                else:
                    # Single file_id
                    test_file_ids.append(graphs.file_id)
    
    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss, test_predictions, test_targets, test_file_ids

def train_epoch_quantiles(model, train_loader, optimizer, criterion, percentiles_tensor, epoch, device, forward_fn):
    """Train for one epoch with quantile regression
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Pinball loss function (already has taus bound via partial)
        percentiles_tensor: Tensor of percentile levels for quantile regression
        epoch: Current epoch number
        device: Device to run on
        forward_fn: Function to forward pass heterogeneous batch through model
    
    Returns:
        Average training loss for the epoch
    """
    import torch.nn.functional as F
    
    model.train()
    print(f"model device: {next(model.parameters()).device}")
    train_loss = 0.0
    num_quantiles = len(percentiles_tensor)
    
    # For plotting every 10 epochs
    train_predictions = []
    train_targets = []
    train_file_ids = []
    
    for batch in train_loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Model outputs logits of shape (B, K) where K is number of quantiles
        logits = forward_fn(model, batch)
        
        # Ensure batch dimension exists (for single-sample batches)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # ----- Monotone parametrization -----
        # logits: [base, inc2_raw, inc3_raw, ...]
        # First quantile is base, rest are cumulative softplus increments
        base = logits[:, 0].unsqueeze(-1)  # (B, 1)
        
        # Build quantiles cumulatively
        pred_quantiles = [base]
        cumulative = base
        
        for i in range(1, num_quantiles):
            increment = F.softplus(logits[:, i]).unsqueeze(-1)  # (B, 1)
            cumulative = cumulative + increment
            pred_quantiles.append(cumulative)
        
        pred_quantiles = torch.cat(pred_quantiles, dim=-1)  # (B, K)
        
        # ----- Build variable-length target list -----
        y_list = split_batch_targets(batch)  # list of length B
        
        # ----- Pinball loss -----
        loss = criterion(pred_quantiles, y_list)
        
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        # Collect predictions and targets for plotting
        if epoch % 10 == 0:
            train_predictions.extend(pred_quantiles.detach().cpu().tolist())
            train_targets.extend([y.detach().cpu().tolist() for y in y_list])
            
            # Handle batched file_ids from PyTorch Geometric DataLoader
            if hasattr(batch, 'file_id'):
                if isinstance(batch.file_id, (list, tuple)):
                    train_file_ids.extend(batch.file_id)
                elif torch.is_tensor(batch.file_id):
                    train_file_ids.extend(batch.file_id.tolist())
                else:
                    # Single file_id
                    train_file_ids.append(batch.file_id)
    
    # Save plot every 10 epochs
    if epoch % 10 == 0:
        # Handle both tensor and list types
        if isinstance(percentiles_tensor, torch.Tensor):
            percentiles_list = percentiles_tensor.cpu().tolist()
        else:
            percentiles_list = percentiles_tensor
        train_plot_path = save_quantile_predictions_plot(
            train_targets, train_predictions, train_file_ids, percentiles_list,
            epoch=epoch, phase="train"
        )
        wandb.log({f"train/quantile_predictions_epoch_{epoch}": wandb.Image(train_plot_path)})
    
    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


def validate_epoch_quantiles(model, val_loader, criterion, percentiles_tensor, epoch, device, forward_fn):
    """Validate for one epoch with quantile regression
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Pinball loss function (already has taus bound via partial)
        percentiles_tensor: Tensor of percentile levels for quantile regression
        epoch: Current epoch number
        device: Device to run on
        forward_fn: Function to forward pass heterogeneous batch through model
    
    Returns:
        Average validation loss for the epoch
    """
    import torch.nn.functional as F
    
    model.eval()
    val_loss = 0.0
    num_quantiles = len(percentiles_tensor)
    
    # For plotting every 10 epochs
    val_predictions = []
    val_targets = []
    val_file_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Model outputs logits of shape (B, K) where K is number of quantiles
            logits = forward_fn(model, batch)
            
            # Ensure batch dimension exists (for single-sample batches)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            # ----- Monotone parametrization -----
            base = logits[:, 0].unsqueeze(-1)  # (B, 1)
            
            # Build quantiles cumulatively
            pred_quantiles = [base]
            cumulative = base
            
            for i in range(1, num_quantiles):
                increment = F.softplus(logits[:, i]).unsqueeze(-1)  # (B, 1)
                cumulative = cumulative + increment
                pred_quantiles.append(cumulative)
            
            pred_quantiles = torch.cat(pred_quantiles, dim=-1)  # (B, K)
            
            # ----- Build variable-length target list -----
            y_list = split_batch_targets(batch)  # list of length B
            
            # ----- Pinball loss -----
            loss = criterion(pred_quantiles, y_list)
            
            val_loss += loss.item()
            
            # Collect predictions and targets for plotting
            if epoch % 10 == 0:
                val_predictions.extend(pred_quantiles.detach().cpu().tolist())
                val_targets.extend([y.detach().cpu().tolist() for y in y_list])
                
                # Handle batched file_ids from PyTorch Geometric DataLoader
                if hasattr(batch, 'file_id'):
                    if isinstance(batch.file_id, (list, tuple)):
                        val_file_ids.extend(batch.file_id)
                    elif torch.is_tensor(batch.file_id):
                        val_file_ids.extend(batch.file_id.tolist())
                    else:
                        # Single file_id
                        val_file_ids.append(batch.file_id)
    
    # Save plot every 10 epochs
    if epoch % 10 == 0:
        # Handle both tensor and list types
        if isinstance(percentiles_tensor, torch.Tensor):
            percentiles_list = percentiles_tensor.cpu().tolist()
        else:
            percentiles_list = percentiles_tensor
        val_plot_path = save_quantile_predictions_plot(
            val_targets, val_predictions, val_file_ids, percentiles_list,
            epoch=epoch, phase="val"
        )
        wandb.log({f"val/quantile_predictions_epoch_{epoch}": wandb.Image(val_plot_path)})
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def evaluate_test_quantiles(model, test_loader, criterion, percentiles_tensor, device, forward_fn):
    """Evaluate on test set with quantile regression
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Pinball loss function (already has taus bound via partial)
        percentiles_tensor: Tensor of percentile levels for quantile regression
        device: Device to run on
        forward_fn: Function to forward pass heterogeneous batch through model
    
    Returns:
        Tuple of (avg_test_loss, test_predictions, test_targets, test_file_ids)
        where test_predictions is a list of quantile arrays
    """
    import torch.nn.functional as F
    
    model.eval()
    test_loss = 0.0
    test_predictions = []  # Will store list of quantile arrays
    test_targets = []  # Will store individual target values (variable length per sample)
    test_file_ids = []
    num_quantiles = len(percentiles_tensor)
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Model outputs logits of shape (B, K) where K is number of quantiles
            logits = forward_fn(model, batch)
            
            # Ensure batch dimension exists (for single-sample batches)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            # ----- Monotone parametrization -----
            base = logits[:, 0].unsqueeze(-1)  # (B, 1)
            
            # Build quantiles cumulatively
            pred_quantiles = [base]
            cumulative = base
            
            for i in range(1, num_quantiles):
                increment = F.softplus(logits[:, i]).unsqueeze(-1)  # (B, 1)
                cumulative = cumulative + increment
                pred_quantiles.append(cumulative)
            
            pred_quantiles = torch.cat(pred_quantiles, dim=-1)  # (B, K)
            
            # ----- Build variable-length target list -----
            y_list = split_batch_targets(batch)  # list of length B
            
            # ----- Pinball loss -----
            loss = criterion(pred_quantiles, y_list)
            
            test_loss += loss.item()
            
            # Store predictions as list of arrays (one array per sample)
            test_predictions.extend(pred_quantiles.detach().cpu().tolist())
            
            # Store targets (one array per sample)
            test_targets.extend([y.detach().cpu().tolist() for y in y_list])
            
            # Handle batched file_ids from PyTorch Geometric DataLoader
            if hasattr(batch, 'file_id'):
                if isinstance(batch.file_id, (list, tuple)):
                    test_file_ids.extend(batch.file_id)
                elif torch.is_tensor(batch.file_id):
                    test_file_ids.extend(batch.file_id.tolist())
                else:
                    # Single file_id
                    test_file_ids.append(batch.file_id)
    
    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss, test_predictions, test_targets, test_file_ids


def train_model(model, train_loader, val_loader, optimizer, criterion, output_transform, 
                config, device, forward_fn, model_save_path, taskType):
    """Main training loop with early stopping
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        output_transform: Transform to apply to output (e.g., Sigmoid)
        config: Configuration object with hyperparameters
        device: Device to run on
        forward_fn: Function to forward pass heterogeneous batch through model
        model_save_path: Path to save the best model
    
    Returns:
        Tuple of (best_val_loss, best_epoch, early_stopped, total_epochs, final_train_loss, final_val_loss)
    """
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    
    print(f"\nStarting training for up to {config.num_epochs} epochs (early stopping after {config.patience} epochs without improvement)...")
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training
        if taskType in [TaskType.QUANTILE_REGRESSION_FEAS, TaskType.QUANTILE_REGRESSION_INFEAS]:
            avg_train_loss = train_epoch_quantiles(
                model, train_loader, optimizer, criterion, config.percentiles_tensor, epoch, device, forward_fn
            )
            # Validation
            avg_val_loss = validate_epoch_quantiles(
                model, val_loader, criterion, config.percentiles_tensor, epoch, device, forward_fn
            )
        else:
            avg_train_loss = train_epoch(
                model, train_loader, optimizer, criterion, output_transform, epoch, device, forward_fn
            )
        
            # Validation
            avg_val_loss = validate_epoch(
                model, val_loader, criterion, output_transform, epoch, device, forward_fn
            )
        
        # Log to WANDB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })
        
        print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Check for improvement and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'config': dict(wandb.config),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_save_path)
            print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f}), (train_loss: {avg_train_loss:.6f})")
        else:
            epochs_without_improvement += 1
            
            # Early stopping check
            if epochs_without_improvement >= config.patience and epoch >= 51:
                print(f"\n🛑 Early stopping triggered! No improvement for {config.patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
                early_stopped = True
                break
    
    return best_val_loss, best_epoch, early_stopped, epoch + 1
