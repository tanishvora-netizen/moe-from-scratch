"""Triton expert computation: every routed expert in two grouped-GEMM launches, one per matmul.

Imports Triton, so the MoE layer only imports this module when cfg.moe_impl is "triton".
Shapes: T tokens, E routed experts, M assignments, d the model width, h the expert hidden size.
"""
import torch
import torch.nn.functional as F

from ..kernels.grouped_gemm import grouped_gemm
from ..routing.assignments import Routing


def fused_experts(x: torch.Tensor, routing: Routing, w_gate_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """x: (T, d), w_gate_up: (E, d, 2h), w_down: (E, h, d) -> (T, d), the gate-weighted sum of expert outputs."""
    hidden = grouped_gemm(x, w_gate_up, routing.offsets, routing.sorted_token)   # (M, 2h), gather fused
    gate, up = hidden.chunk(2, dim=-1)
    expert_out = grouped_gemm(F.silu(gate) * up, w_down, routing.offsets)        # (M, d)
    # Accumulate in fp32: a token's k expert outputs are summed, and low-precision adds lose bits.
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    out.index_add_(0, routing.sorted_token, expert_out.float() * routing.sorted_weight[:, None])
    return out.to(x.dtype)
