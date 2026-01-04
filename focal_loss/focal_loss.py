import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss as described in Lin et al. 2017.
    Supports optional per‑class weighting (alpha) and focusing parameter gamma.
    Args:
        gamma (float): focusing parameter, default 2.0.
        alpha (float|list|Tensor, optional): weighting factor for each class.
            If a single float is provided, the same weight is applied to all classes.
            If a list/Tensor is provided, it must have length = num_classes.
        reduction (str): 'mean' or 'sum' (default 'mean').
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                # scalar weight for all classes
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        Args:
            logits: raw model outputs, shape (N, C).
            targets: ground‑truth class indices, shape (N,).
        Returns:
            loss scalar.
        """
        # Convert logits to probabilities
        prob = F.softmax(logits, dim=1)
        # Gather the probabilities of the true class
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        log_pt = torch.log(pt + 1e-8)
        # Compute the modulating factor
        mod_factor = (1 - pt) ** self.gamma
        loss = -mod_factor * log_pt
        # Apply alpha if provided
        if self.alpha is not None:
            if self.alpha.numel() == 1:
                loss = loss * self.alpha
            else:
                # per‑class alpha
                alpha_t = self.alpha.to(logits.device).gather(0, targets)
                loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
