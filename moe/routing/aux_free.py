"""DeepSeek-V3 auxiliary-loss-free balancing (Wang et al. 2024; DeepSeek-V3): sigmoid affinities and a
per-expert bias that only affects top-k selection, nudged after every optimizer step toward
under-loaded experts.
"""
import torch

from ..config import ModelConfig
from .assignments import Routing, token_choice_routing
from .base import Router


class AuxFreeRouter(Router):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.bias_update_rate = cfg.bias_update_rate
        # Buffers, not parameters: updated by a rule, never by gradients.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))
        self.register_buffer("load_accum", torch.zeros(cfg.n_routed_experts))   # assignments since update_bias

    def forward(self, x: torch.Tensor, causal: bool = False) -> Routing:
        scores = torch.sigmoid(self.logits(x))
        _, top_experts = (scores + self.expert_bias).topk(self.top_k, dim=-1)   # bias changes which experts win
        top_weights = scores.gather(-1, top_experts)                           # gates still use the unbiased scores
        routing = token_choice_routing(top_experts, top_weights, self.n_experts)
        if self.training:
            self.load_accum += routing.counts.float()
        return routing

    @torch.no_grad()
    def update_bias(self) -> None:
        """DeepSeek-V3 rule, called once per optimizer step: b_e += gamma * sign(mean_load - load_e)."""
        error = self.load_accum.mean() - self.load_accum   # > 0 for under-loaded experts
        self.expert_bias += self.bias_update_rate * torch.sign(error)
        self.load_accum.zero_()
