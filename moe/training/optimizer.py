"""The AdamW optimizer with its parameter groups, and the learning-rate schedule."""
import argparse
import math

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


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup over args.warmup steps, then cosine decay to args.min_lr_frac * args.lr at args.steps."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)
