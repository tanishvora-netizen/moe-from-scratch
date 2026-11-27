"""The DeepSeekMoE feed-forward layer (Dai et al. 2024), with shared experts on every token plus k routed experts each.
Shapes: T tokens (batch * seq), E routed experts, k experts per token, d the model width, h the expert hidden size.
"""
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..model.swiglu import SwiGLU
from ..routing import build_router
from ..routing.assignments import Routing
from .naive import naive_experts


class MoELayer(nn.Module):
    """y = shared(x) + sum over the token's routed experts e of gate_e * expert_e(x).

    1. The router assigns each token to experts, in the Routing layout.
    2. Each routed expert runs a small SwiGLU on its tokens, in the Triton kernel or a Python loop.
    3. Each expert output is scaled by its gate weight and added to its token's row.
    4. The shared experts run on every token, and their output is added.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        E, d, h = cfg.n_routed_experts, cfg.d_model, cfg.expert_hidden
        self.n_experts, self.top_k, self.impl = E, cfg.top_k, cfg.moe_impl
        self.router = build_router(cfg)
        # Stacked by expert id, the layout grouped_gemm indexes. The naive path reads the same tensors.
        self.w_gate_up = nn.Parameter(torch.empty(E, d, 2 * h))   # gate and up projections in one matrix
        self.w_down = nn.Parameter(torch.empty(E, h, d))
        # n_shared_experts experts of width h behave as one SwiGLU of width n_shared_experts * h.
        self.shared = SwiGLU(d, cfg.n_shared_experts * h) if cfg.n_shared_experts > 0 else None

    def reset_parameters(self, std: float, residual_out_std: float) -> None:
        """Draws the stacked routed weights, which MoETransformer._init_weights cannot reach as nn.Linear modules."""
        nn.init.normal_(self.w_gate_up, std=std)
        nn.init.normal_(self.w_down, std=residual_out_std)

    def forward(self, x: torch.Tensor, causal_routing: bool = False) -> tuple[torch.Tensor, Routing]:
        """Returns (y, routing). x is (..., d), and y has the same shape."""
        shape = x.shape
        x = x.reshape(-1, shape[-1])                       # (T, d)
        routing = self.router(x, causal=causal_routing)
        if self.impl == "triton":
            from .fused import fused_experts  # imported here so the naive path works without Triton
            out = fused_experts(x, routing, self.w_gate_up, self.w_down)
        else:
            out = naive_experts(x, routing, self.w_gate_up, self.w_down)
        if self.shared is not None:
            out = out + self.shared(x)
        return out.view(shape), routing
