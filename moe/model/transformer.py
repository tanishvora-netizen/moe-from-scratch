"""MoETransformer, the decoder-only language model: embedding, blocks, LM head, loss and sampling.

With cfg.ffn_type="dense" the same class is the matched dense baseline.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..experts.layer import MoELayer
from ..routing.aux_free import AuxFreeRouter
from .block import Block
from .norm import RMSNorm
from .rope import precompute_rope


class MoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, layer_idx) for layer_idx in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying
        cos, sin = precompute_rope(cfg.d_model // cfg.n_heads, cfg.seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        out_std = std / math.sqrt(2 * self.cfg.n_layers)   # GPT-2 style scaling for residual outputs
        # named_modules() yields each MoELayer before its children, which fixes the RNG order.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_out = name.endswith("attn.proj") or name.endswith("w_down")
                nn.init.normal_(module.weight, std=out_std if is_out else std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)
            elif isinstance(module, MoELayer):
                module.reset_parameters(std, out_std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, causal_routing: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """idx, targets: (batch, seq) token ids -> (logits, loss, info).

        loss is the cross-entropy, plus the balance loss when cfg.router is "aux_loss"; None without targets.
        info holds, detached for logging, "aux" (mean balance loss per MoE layer) and "ce" (with targets),
        and "routing", each MoE layer's Routing in layer order.
        causal_routing=True makes expert-choice layers route each token on its own (see routing.expert_choice).
        """
        seq = idx.size(1)
        if seq > self.cfg.seq_len:
            raise ValueError(f"sequence length {seq} > seq_len {self.cfg.seq_len}")
        x = self.tok_emb(idx)
        routings = []
        for block in self.blocks:
            x, routing = block(x, self.rope_cos, self.rope_sin, causal_routing)
            if routing is not None:
                routings.append(routing)
        logits = self.lm_head(self.norm_f(x))

        if routings:
            aux_total = torch.stack([routing.aux_loss for routing in routings]).sum()
        else:
            aux_total = logits.new_zeros(()).float()
        info = {"aux": aux_total.detach() / max(1, len(routings)), "routing": routings}
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce
            if self.cfg.ffn_type == "moe" and self.cfg.router == "aux_loss":
                loss = ce + self.cfg.aux_loss_coef * aux_total
            info["ce"] = ce.detach()
        return logits, loss, info

    def moe_layers(self) -> list[MoELayer]:
        return [module for module in self.modules() if isinstance(module, MoELayer)]

    def update_router_biases(self) -> None:
        """Applies the aux-free bias rule once per optimizer step. Other routers have no bias."""
        for module in self.modules():
            if isinstance(module, AuxFreeRouter):
                module.update_bias()

    def num_params(self) -> tuple[int, int]:
        """(total, active-per-token). Shared (tied) params are counted once."""
        total = sum(param.numel() for param in self.parameters())
        inactive = 0
        for layer in self.moe_layers():
            routed = layer.w_gate_up.numel() + layer.w_down.numel()
            inactive += routed * (1 - layer.top_k / layer.n_experts)
        return total, int(total - inactive)

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50
    ) -> torch.Tensor:
        """Samples max_new_tokens after idx: (batch, seq). Routing is causal, so no token sees the future."""
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -self.cfg.seq_len:], causal_routing=True)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = -float("inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
