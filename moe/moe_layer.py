"""
DeepSeek-MoE style layer:  y = SharedExpert(x) + sum_{e in topk(x)} g_e(x) * Expert_e(x)

Routed expert weights are stored STACKED, (E, d, 2h) and (E, h, d), so the Triton
grouped GEMM can index them by expert id. The naive path uses the same tensors,
which makes the two implementations exactly comparable (and checkpoint-compatible).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SwiGLU
from .router import Router, Routing


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, d, h = cfg.n_routed_experts, cfg.d_model, cfg.expert_hidden
        self.n_experts, self.top_k, self.impl = E, cfg.top_k, cfg.moe_impl
        self.router = Router(cfg)
        self.w_gate_up = nn.Parameter(torch.empty(E, d, 2 * h))   # gate & up fused
        self.w_down = nn.Parameter(torch.empty(E, h, d))
        self.shared = SwiGLU(d, cfg.n_shared_experts * h) if cfg.n_shared_experts > 0 else None

        # stats for logging / analysis (filled every forward)
        self.record = False          # set True to keep per-token assignments (analyze_experts.py)
        self.last_counts = None
        self.last_dropped = None
        self.last_assign = None

    def reset_parameters(self, std: float, out_std: float):
        nn.init.normal_(self.w_gate_up, std=std)
        nn.init.normal_(self.w_down, std=out_std)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])                       # (T, d), T = batch * seq
        r = self.router(x)
        if self.impl == "triton":
            y = self._experts_triton(x, r)
        else:
            y = self._experts_naive(x, r)
        if self.shared is not None:
            y = y + self.shared(x)
        self._record_stats(r, x.size(0))
        return y.view(shape), r.aux_loss

    # ------------------------------------------------------------------------------------------
    def _experts_triton(self, x, r: Routing):
        from .kernels import grouped_gemm   # lazy import: the naive path works without Triton
        h = grouped_gemm(x, self.w_gate_up, r.offsets, r.sorted_token)   # (M, 2h), gather fused
        g, u = h.chunk(2, dim=-1)
        a = F.silu(g) * u
        y = grouped_gemm(a, self.w_down, r.offsets)                      # (M, d)
        return self._combine(y, r, x)

    def _experts_naive(self, x, r: Routing):
        """Reference: Python loop over experts, one gather + 2 matmuls + scatter each."""
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        offs = r.offsets.tolist()                          # host sync (part of why this is slow)
        for e in range(self.n_experts):
            s, t = offs[e], offs[e + 1]
            if s == t:
                continue
            tok = r.sorted_token[s:t]
            g, u = (x[tok] @ self.w_gate_up[e]).chunk(2, dim=-1)
            ye = (F.silu(g) * u) @ self.w_down[e]
            out.index_add_(0, tok, ye.float() * r.sorted_weight[s:t, None])
        return out.to(x.dtype)

    @staticmethod
    def _combine(y, r: Routing, x):
        """Weight each expert output by its gate and scatter-add back to its token."""
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        out.index_add_(0, r.sorted_token, y.float() * r.sorted_weight[:, None])
        return out.to(x.dtype)

    @torch.no_grad()
    def _record_stats(self, r: Routing, n_tokens: int):
        self.last_counts = r.counts.detach().float()
        if self.router.strategy == "expert_choice" and self.training:
            covered = torch.bincount(r.sorted_token, minlength=n_tokens)
            self.last_dropped = (covered == 0).float().mean()   # tokens that got ZERO routed experts
        else:
            self.last_dropped = None
        if self.record:
            self.last_assign = (r.sorted_token.detach(), r.sorted_expert.detach())
