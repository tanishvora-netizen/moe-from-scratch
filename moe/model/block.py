"""One pre-norm transformer block: attention, then an MoE or dense SwiGLU feed-forward."""
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..experts.layer import MoELayer
from ..routing.assignments import Routing
from .attention import CausalSelfAttention
from .norm import RMSNorm
from .swiglu import SwiGLU


class Block(nn.Module):
    """x + attn(norm(x)), then x + ffn(norm(x)). The first cfg.first_k_dense blocks use a dense FFN."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.is_moe = cfg.ffn_type == "moe" and layer_idx >= cfg.first_k_dense
        self.ffn = MoELayer(cfg) if self.is_moe else SwiGLU(cfg.d_model, cfg.dense_hidden)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, causal_routing: bool = False
    ) -> tuple[torch.Tensor, Routing | None]:
        """x: (batch, seq, d_model) -> (same shape, the MoE layer's Routing or None for a dense block)."""
        x = x + self.attn(self.norm1(x), cos, sin)
        if not self.is_moe:
            return x + self.ffn(self.norm2(x)), None
        out, routing = self.ffn(self.norm2(x), causal_routing)
        return x + out, routing
