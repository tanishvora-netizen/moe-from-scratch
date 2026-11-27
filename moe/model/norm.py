"""RMSNorm, the pre-norm used in every transformer block and before the LM head."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight over the last dimension, computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)
