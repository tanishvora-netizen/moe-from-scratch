"""The Triton expert path, which covers every routed expert in two grouped-GEMM launches. It imports Triton,
so MoELayer imports this module only when moe_impl is "triton".
Shapes: T tokens, E routed experts, M assignments, d the model width, h the expert hidden size.
"""
import torch
import torch.nn.functional as F

from ..kernels.grouped_gemm import grouped_gemm
from ..routing.assignments import Routing


def fused_experts(x: torch.Tensor, routing: Routing, w_gate_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Returns the (T, d) gate-weighted sum of each token's expert outputs.

    x is (T, d), w_gate_up is (E, d, 2h) and w_down is (E, h, d).
    """
    hidden = grouped_gemm(x, w_gate_up, routing.offsets, routing.sorted_token)   # (M, 2h), gather fused in
    gate, up = hidden.chunk(2, dim=-1)
    expert_out = grouped_gemm(F.silu(gate) * up, w_down, routing.offsets)        # (M, d)
    # Sum in fp32, because adding a token's expert outputs in bf16 loses precision.
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    out.index_add_(0, routing.sorted_token, expert_out.float() * routing.sorted_weight[:, None])
    return out.to(x.dtype)
