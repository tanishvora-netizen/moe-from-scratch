"""The reference expert path, a Python loop that does one gather, two matmuls and one scatter per expert.
Shapes: T tokens, E routed experts, d the model width, h the expert hidden size.
"""
import torch
import torch.nn.functional as F

from ..routing.assignments import Routing


def naive_experts(x: torch.Tensor, routing: Routing, w_gate_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Returns the (T, d) gate-weighted sum of each token's expert outputs, the result fused_experts must match.

    x is (T, d), w_gate_up is (E, d, 2h) and w_down is (E, h, d).
    """
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    offsets = routing.offsets.tolist()                 # a host sync, one reason this path is slow
    for expert in range(w_gate_up.size(0)):
        start, end = offsets[expert], offsets[expert + 1]
        if start == end:
            continue
        tokens = routing.sorted_token[start:end]
        gate, up = (x[tokens] @ w_gate_up[expert]).chunk(2, dim=-1)
        expert_out = (F.silu(gate) * up) @ w_down[expert]
        out.index_add_(0, tokens, expert_out.float() * routing.sorted_weight[start:end, None])
    return out.to(x.dtype)
