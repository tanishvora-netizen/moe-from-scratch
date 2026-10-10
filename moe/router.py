"""
Routers. All three strategies return the same "sorted" layout that the grouped-GEMM
kernel consumes, so the expert computation is identical across the ablation.

  aux_loss       token-choice top-k over a softmax + Switch/GShard balance loss
                 (Fedus et al. 2021; Lepikhin et al. 2020)
  aux_free       DeepSeek-V3 auxiliary-loss-free balancing: sigmoid affinities, a
                 per-expert bias added ONLY for top-k selection, nudged after every
                 step toward under-loaded experts (Wang et al. 2024; DeepSeek-V3)
  expert_choice  each expert picks its top-C tokens (Zhou et al. 2022). Perfectly
                 balanced by construction, but NOT causal for a decoder LM: whether a
                 token is picked depends on other tokens in the batch, including future
                 ones. See eval_mode below and the README.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Routing:
    sorted_token: torch.Tensor   # (M,)  token index of each assignment, grouped by expert
    sorted_expert: torch.Tensor  # (M,)  expert id of each assignment (non-decreasing)
    sorted_weight: torch.Tensor  # (M,)  gate weight (differentiable)
    offsets: torch.Tensor        # (E+1,) expert e owns rows [offsets[e], offsets[e+1])
    counts: torch.Tensor         # (E,)  assignments per expert
    aux_loss: torch.Tensor       # scalar (zero unless router == "aux_loss")


class Router(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.strategy = cfg.router
        self.capacity_factor = cfg.capacity_factor
        self.bias_update_rate = cfg.bias_update_rate
        self.gate = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)
        # Aux-loss-free state. Buffers, not parameters: updated by a rule, never by gradients.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))
        self.register_buffer("load_accum", torch.zeros(cfg.n_routed_experts))
        # Set to "token_choice" to evaluate an expert-choice model causally (top-k per token).
        self.eval_mode = None

    def forward(self, x: torch.Tensor) -> Routing:
        # Router math in fp32 even under autocast (small, and precision matters for top-k).
        ac_device = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=ac_device, enabled=False):
            logits = F.linear(x.float(), self.gate.weight.float())   # (T, E)

        strategy = self.strategy
        if strategy == "expert_choice" and not self.training and self.eval_mode == "token_choice":
            strategy = "softmax_topk"
        if strategy == "expert_choice":
            return self._expert_choice(logits)
        return self._token_choice(logits, strategy)

    # ------------------------------------------------------------------------------------------
    def _token_choice(self, logits, strategy):
        T, E, k = logits.size(0), self.n_experts, self.top_k
        if strategy == "aux_free":
            scores = torch.sigmoid(logits)
            _, topk_idx = (scores + self.expert_bias).topk(k, dim=-1)   # bias affects WHICH experts...
            topk_w = scores.gather(-1, topk_idx)                        # ...but not the gate values
        else:
            probs = logits.softmax(dim=-1)
            topk_w, topk_idx = probs.topk(k, dim=-1)
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-9)

        flat = topk_idx.reshape(-1)                                     # (T*k,)
        counts = torch.bincount(flat, minlength=E)

        aux = logits.new_zeros(())
        if strategy == "aux_loss":
            f = counts.float() / (T * k)          # fraction of assignments per expert (no grad)
            p = probs.mean(0)                     # mean router prob per expert (has grad)
            aux = E * (f * p).sum()               # = 1.0 when perfectly balanced
        if strategy == "aux_free" and self.training:
            with torch.no_grad():
                self.load_accum += counts.float()

        sorted_expert, order = torch.sort(flat, stable=True)
        offsets = F.pad(torch.cumsum(counts, 0), (1, 0))
        return Routing(
            sorted_token=order // k,
            sorted_expert=sorted_expert,
            sorted_weight=topk_w.reshape(-1)[order],
            offsets=offsets,
            counts=counts,
            aux_loss=aux,
        )

    def _expert_choice(self, logits):
        T, E = logits.shape
        probs = logits.softmax(dim=-1)                                   # (T, E)
        cap = int(self.capacity_factor * T * self.top_k / E)
        cap = max(1, min(cap, T))
        gate_w, tok = probs.t().topk(cap, dim=-1)                        # (E, cap): each expert picks tokens
        device = logits.device
        return Routing(
            sorted_token=tok.reshape(-1),
            sorted_expert=torch.arange(E, device=device).repeat_interleave(cap),
            sorted_weight=gate_w.reshape(-1),
            offsets=torch.arange(E + 1, device=device) * cap,
            counts=torch.full((E,), cap, device=device, dtype=torch.long),
            aux_loss=logits.new_zeros(()),
        )

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def update_bias(self):
        """DeepSeek-V3 rule, called once per optimizer step: b_e += gamma * sign(mean_load - load_e)."""
        if self.strategy != "aux_free":
            return
        err = self.load_accum.mean() - self.load_accum   # > 0 for under-loaded experts
        self.expert_bias += self.bias_update_rate * torch.sign(err)
        self.load_accum.zero_()
