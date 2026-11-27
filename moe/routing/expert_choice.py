"""Expert-choice routing (Zhou et al. 2022): each expert picks its top-C tokens.

Balanced by construction, but not causal for a decoder LM: whether a token is picked depends on
the other tokens in the batch, future ones included. With causal=True each token picks its own
top k instead, which is how expert-choice models are evaluated and sampled from.

Shapes: T tokens, E routed experts, k experts per token, C the per-expert capacity.
"""
import torch

from ..config import ModelConfig
from .assignments import Routing, token_choice_routing
from .base import Router


class ExpertChoiceRouter(Router):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.capacity_factor = cfg.capacity_factor

    def forward(self, x: torch.Tensor, causal: bool = False) -> Routing:
        logits = self.logits(x)
        probs = logits.softmax(dim=-1)                                       # (T, E)
        if causal:
            top_weights, top_experts = probs.topk(self.top_k, dim=-1)
            return token_choice_routing(top_experts, top_weights, self.n_experts)

        # C = capacity_factor * T * k / E: at capacity_factor 1 the experts take as many assignments
        # in total as token choice makes.
        T, E = logits.shape
        capacity = int(self.capacity_factor * T * self.top_k / E)
        capacity = max(1, min(capacity, T))
        gate_weights, tokens = probs.t().topk(capacity, dim=-1)              # (E, C): each expert picks tokens
        device = logits.device
        return Routing(
            sorted_token=tokens.reshape(-1),
            sorted_expert=torch.arange(E, device=device).repeat_interleave(capacity),
            sorted_weight=gate_weights.reshape(-1),
            offsets=torch.arange(E + 1, device=device) * capacity,
            counts=torch.full((E,), capacity, device=device, dtype=torch.long),
            aux_loss=logits.new_zeros(()),
            n_tokens=T,
            can_drop_tokens=True,
        )
