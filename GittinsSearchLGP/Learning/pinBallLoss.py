import torch

def pinball_loss_varlen(pred_quantiles, targets_list, taus, reduction="mean"):
    """
    Pinball (quantile) loss for batched graphs with variable #samples per graph.

    Args
    ----
    pred_quantiles : torch.Tensor, shape (B, K)
        Model predictions: K quantiles for each of B graphs.
    targets_list   : list of torch.Tensor
        List of length B. Each element is a tensor of shape (M_i,)
        containing the target samples for graph i.
    taus           : torch.Tensor, shape (K,)
        Quantile levels in (0, 1), e.g. tensor([0.1, 0.5, 0.9]).
    reduction      : str, "mean" | "sum" | "none"
        How to reduce the loss.

    Returns
    -------
    loss : torch.Tensor
        If reduction="none": shape (sum_i M_i, K)
        Else: scalar tensor.
    """
    device = pred_quantiles.device
    taus = taus.to(device)
    
    B, K = pred_quantiles.shape
    assert len(targets_list) == B, "targets_list must have length B"
    assert taus.dim() == 1 and taus.numel() == K, "taus must be (K,)"

    # Build big tensors of predictions and targets by concatenation
    pred_expanded = []
    target_expanded = []

    for i, y_i in enumerate(targets_list):
        # y_i: (M_i,)
        y_i = y_i.to(device)

        M_i = y_i.shape[0]
        assert M_i > 0, "Each graph must have at least one target sample"

        # Repeat the same predicted quantiles M_i times: (M_i, K)
        p_i = pred_quantiles[i].unsqueeze(0).expand(M_i, K)

        pred_expanded.append(p_i)
        target_expanded.append(y_i.unsqueeze(1))  # (M_i, 1)

    # Concatenate over all graphs
    pred_all = torch.cat(pred_expanded, dim=0)     # (N_total, K)
    y_all = torch.cat(target_expanded, dim=0)      # (N_total, 1)
    # Broadcast y_all against pred_all: (N_total, K)
    diff = y_all - pred_all

    # taus -> (1, K) for broadcasting against (N_total, K)
    taus_ = taus.view(1, K)

    # pinball loss: max(τ * diff, (τ - 1) * diff)
    loss = torch.maximum(taus_ * diff, (taus_ - 1) * diff)  # (N_total, K)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")