"""Reference expert computation: a Python loop over experts, one gather, two matmuls and a scatter each.

Shapes: T tokens, E routed experts, d the model width, h the expert hidden size.
"""
import torch
import torch.nn.functional as F

from ..routing.assignments import Routing


def naive_experts(x: torch.Tensor, routing: Routing, w_gate_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """x: (T, d), w_gate_up: (E, d, 2h), w_down: (E, h, d) -> (T, d), the gate-weighted sum of expert outputs."""
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    offsets = routing.offsets.tolist()                 # host sync (part of why this is slow)
    for expert in range(w_gate_up.size(0)):
        start, end = offsets[expert], offsets[expert + 1]
        if start == end:
            continue
        tokens = routing.sorted_token[start:end]
        gate, up = (x[tokens] @ w_gate_up[expert]).chunk(2, dim=-1)
        expert_out = (F.silu(gate) * up) @ w_down[expert]
        out.index_add_(0, tokens, expert_out.float() * routing.sorted_weight[start:end, None])
    return out.to(x.dtype)
