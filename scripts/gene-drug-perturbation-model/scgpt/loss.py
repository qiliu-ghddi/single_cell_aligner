import torch
import torch.nn.functional as F
import numpy as np

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()
def weighted_mse_loss(control: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Custom weighted MSE loss.

    Args:
    - control (torch.Tensor): Control tensor.
    - target (torch.Tensor): Perturbed tensor.
    - output (torch.Tensor): Model Output tensor.

    Returns:
    - torch.Tensor: The computed loss value.
    """
    # Convert tensors to float if necessary
    if control.dtype != torch.float32:
        control = control.float()
    if output.dtype != torch.float32:
        output = output.float()
    if target.dtype != torch.float32:
        target = target.float()
    
    mse_loss = F.mse_loss(output, target, reduction="mean")
    weights = torch.sigmoid(torch.abs(target - control))
    weighted_mse = weights * (target - output)**2
    loss = torch.mean(weighted_mse) + mse_loss
    
    return loss

def pertmean_mse_loss(
    perd: torch.Tensor, input_gene_ids, perts ,pertmean
):
    """
    Compute the masked MSE loss between input and target.
    """
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(perd.device)
    for p in perts:

        losses = losses + F.mse_loss(perd , pertmean[p][input_gene_ids], reduction="sum")
    
    return losses / len(perts)



def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
