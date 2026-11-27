"""The AdamW optimizer and its parameter groups."""
import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module, weight_decay: float, lr: float, device_type: str, betas: tuple[float, float] = (0.9, 0.95)
) -> torch.optim.AdamW:
    """AdamW that decays matrices and embeddings (dim >= 2) but not norm weights or other vectors."""
    decay = [param for param in model.parameters() if param.requires_grad and param.dim() >= 2]
    no_decay = [param for param in model.parameters() if param.requires_grad and param.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"fused": True} if device_type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)
