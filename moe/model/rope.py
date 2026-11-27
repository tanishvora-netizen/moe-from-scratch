"""Rotary position embeddings (Su et al. 2021). precompute_rope builds the cos and sin tables, and apply_rope
rotates queries and keys with them. Shapes: T is the sequence length.
"""
import torch


def precompute_rope(head_dim: int, seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each (seq_len, head_dim / 2), with one rotation angle per position and channel pair."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates the channel pairs (x0, x1), (x2, x3) and so on of x, which is (batch, heads, T, head_dim)."""
    T = x.size(2)
    cos, sin = cos[None, None, :T], sin[None, None, :T]
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return out.type_as(x)
