import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import CausalSelfAttention, RMSNorm, SwiGLU, precompute_rope
from .moe_layer import MoE
from .router import Router


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.is_moe = cfg.ffn_type == "moe" and layer_idx >= cfg.first_k_dense
        self.ffn = MoE(cfg) if self.is_moe else SwiGLU(cfg.d_model, cfg.dense_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        h = self.norm2(x)
        if self.is_moe:
            y, aux = self.ffn(h)
        else:
            y, aux = self.ffn(h), None
        return x + y, aux


class MoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying
        cos, sin = precompute_rope(cfg.d_model // cfg.n_heads, cfg.seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        out_std = std / math.sqrt(2 * self.cfg.n_layers)   # GPT-2 style scaling for residual outputs
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                is_out = name.endswith("attn.proj") or name.endswith("w_down")
                nn.init.normal_(m.weight, std=out_std if is_out else std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
            elif isinstance(m, MoE):
                m.reset_parameters(std, out_std)

    # ------------------------------------------------------------------------------------------
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"sequence length {T} > seq_len {self.cfg.seq_len}"
        x = self.tok_emb(idx)
        aux_terms = []
        for blk in self.blocks:
            x, aux = blk(x, self.rope_cos, self.rope_sin)
            if aux is not None:
                aux_terms.append(aux)
        logits = self.lm_head(self.norm_f(x))

        aux_total = torch.stack(aux_terms).sum() if aux_terms else logits.new_zeros(()).float()
        info = {"aux": aux_total.detach() / max(1, len(aux_terms))}
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce
            if self.cfg.ffn_type == "moe" and self.cfg.router == "aux_loss":
                loss = ce + self.cfg.aux_loss_coef * aux_total
            info["ce"] = ce.detach()
        return logits, loss, info

    # ------------------------------------------------------------------------------------------
    def moe_layers(self):
        return [m for m in self.modules() if isinstance(m, MoE)]

    def update_router_biases(self):
        for m in self.modules():
            if isinstance(m, Router):
                m.update_bias()

    def set_router_eval_mode(self, mode):
        """mode=None (default routing) or "token_choice" (causal eval for expert-choice models)."""
        for m in self.modules():
            if isinstance(m, Router):
                m.eval_mode = mode

    @torch.no_grad()
    def routing_stats(self):
        """MaxVio = (max_load - mean_load) / mean_load, averaged over MoE layers (0 = perfect)."""
        maxvio, loads, dropped = [], [], []
        for m in self.moe_layers():
            if m.last_counts is None:
                continue
            c = m.last_counts
            maxvio.append(((c.max() - c.mean()) / c.mean()).item())
            loads.append([round(v, 4) for v in (c / c.sum()).tolist()])
            if m.last_dropped is not None:
                dropped.append(float(m.last_dropped))
        if not maxvio:
            return {}
        out = {"maxvio": sum(maxvio) / len(maxvio), "maxvio_per_layer": maxvio, "loads": loads}
        if dropped:
            out["dropped_frac"] = sum(dropped) / len(dropped)
        return out

    def num_params(self):
        """(total, active-per-token). Shared (tied) params are counted once."""
        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for m in self.moe_layers():
            routed = m.w_gate_up.numel() + m.w_down.numel()
            inactive += routed * (1 - m.top_k / m.n_experts)
        return total, int(total - inactive)

    def configure_optimizer(self, weight_decay, lr, device_type, betas=(0.9, 0.95)):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        kwargs = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -self.cfg.seq_len:])
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def load_checkpoint(path, device="cpu", impl=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    if impl is not None:
        cfg_dict["moe_impl"] = impl
    cfg = ModelConfig(**cfg_dict)
    model = MoETransformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg, ckpt.get("args", {})
