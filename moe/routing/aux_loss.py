"""Token-choice top-k over a softmax, balanced by the Switch/GShard auxiliary loss
(Fedus et al. 2021; Lepikhin et al. 2020).
"""
from dataclasses import replace

import torch

from .assignments import Routing, token_choice_routing
from .base import Router


class AuxLossRouter(Router):
    def forward(self, x: torch.Tensor, causal: bool = False) -> Routing:
        logits = self.logits(x)
        probs = logits.softmax(dim=-1)
        top_weights, top_experts = probs.topk(self.top_k, dim=-1)
        routing = token_choice_routing(top_experts, top_weights, self.n_experts)

        # E * sum_e f_e * P_e, which is 1.0 when perfectly balanced and E / k when every token
        # picks the same k experts. f has no gradient; P carries it to the gate.
        assignment_frac = routing.counts.float() / (routing.n_tokens * self.top_k)   # f_e
        mean_prob = probs.mean(0)                                                     # P_e
        return replace(routing, aux_loss=self.n_experts * (assignment_frac * mean_prob).sum())
