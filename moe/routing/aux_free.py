"""Auxiliary-loss-free balancing (Wang et al. 2024), as used in DeepSeek-V3 (DeepSeek-AI 2024). Experts are picked
by sigmoid score plus a per-expert bias, and update_bias moves the bias toward under-loaded experts.
"""
import torch

from ..config import ModelConfig
from .assignments import Routing, token_choice_routing
from .base import Router


class AuxFreeRouter(Router):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.bias_update_rate = cfg.bias_update_rate
        # Buffers, not parameters, so only forward and update_bias change them.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))
        self.register_buffer("load_accum", torch.zeros(cfg.n_routed_experts))   # assignments since update_bias

    def forward(self, x: torch.Tensor, causal: bool = False) -> Routing:
        scores = torch.sigmoid(self.logits(x))
        selection_scores = scores + self.expert_bias
        _, top_experts = selection_scores.topk(self.top_k, dim=-1)
        top_weights = scores.gather(-1, top_experts)   # gates use the scores without the bias
        routing = token_choice_routing(top_experts, top_weights, self.n_experts)
        if self.training:
            self.load_accum += routing.counts.float()
        return routing

    @torch.no_grad()
    def update_bias(self) -> None:
        """The DeepSeek-V3 update, called once per optimizer step: b_e += gamma * sign(mean_load - load_e)."""
        underload = self.load_accum.mean() - self.load_accum
        self.expert_bias += self.bias_update_rate * torch.sign(underload)
        self.load_accum.zero_()
