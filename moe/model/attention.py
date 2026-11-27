"""Causal multi-head self-attention with rotary embeddings on queries and keys.
Shapes: T is the sequence length, d the model width.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x and the result are (batch, T, d). cos and sin are the precompute_rope tables."""
        batch, T, d = x.shape
        query, key, value = self.qkv(x).split(d, dim=-1)
        query = query.view(batch, T, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, T, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, T, self.n_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        out = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, T, d)
        return self.proj(out)
