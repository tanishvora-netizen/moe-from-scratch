"""DeepSeek-MoE style feed-forward layer:  y = SharedExperts(x) + sum_{e in topk(x)} g_e(x) * Expert_e(x)

Routed expert weights are stacked into (E, d, 2h) and (E, h, d) tensors so the Triton grouped GEMM
can index them by expert id. The naive path uses the same tensors, so the two are exactly comparable.
Shapes: T tokens, E routed experts, d the model width, h the expert hidden size.
"""
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..model.swiglu import SwiGLU
from ..routing import build_router
from ..routing.assignments import Routing
from .naive import naive_experts


class MoELayer(nn.Module):
    """Mixture-of-Experts feed-forward layer.

    1. the router assigns each token to experts (see routing.assignments.Routing for the layout)
    2. every routed expert runs a small SwiGLU on its tokens (Triton kernel or Python loop)
    3. each expert's output is scaled by its gate weight and added back to its token
    4. the shared experts run on every token and are added on top
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        E, d, h = cfg.n_routed_experts, cfg.d_model, cfg.expert_hidden
        self.n_experts, self.top_k, self.impl = E, cfg.top_k, cfg.moe_impl
        self.router = build_router(cfg)
        self.w_gate_up = nn.Parameter(torch.empty(E, d, 2 * h))   # gate and up fused
        self.w_down = nn.Parameter(torch.empty(E, h, d))
        # n_shared_experts experts of width h behave as one SwiGLU of width n_shared_experts * h.
        self.shared = SwiGLU(d, cfg.n_shared_experts * h) if cfg.n_shared_experts > 0 else None

    def reset_parameters(self, std: float, residual_out_std: float) -> None:
        nn.init.normal_(self.w_gate_up, std=std)
        nn.init.normal_(self.w_down, std=residual_out_std)

    def forward(self, x: torch.Tensor, causal_routing: bool = False) -> tuple[torch.Tensor, Routing]:
        """x: (..., d) -> (y with the same shape as x, the Routing used)."""
        shape = x.shape
        x = x.reshape(-1, shape[-1])                       # (T, d), T = batch * seq
        routing = self.router(x, causal=causal_routing)
        if self.impl == "triton":
            from .fused import fused_experts  # lazy: the naive path works without Triton
            out = fused_experts(x, routing, self.w_gate_up, self.w_down)
        else:
            out = naive_experts(x, routing, self.w_gate_up, self.w_down)
        if self.shared is not None:
            out = out + self.shared(x)
        return out.view(shape), routing
