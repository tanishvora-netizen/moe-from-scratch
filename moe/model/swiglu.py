"""SwiGLU feed-forward (Shazeer 2020), used as the dense FFN and as the MoE layer's shared experts."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """down(silu(x W_gate) * (x W_up)). Gate and up are fused into one matmul."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)
