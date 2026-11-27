"""The Router base class: the gate projection and fp32 router logits that every routing strategy shares.

Shapes: T tokens, E routed experts, d the model width.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .assignments import Routing


class Router(nn.Module):
    """Scores every token against every routed expert. Subclasses turn the scores into a Routing."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.gate = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, d) -> (T, E) fp32 scores."""
        # fp32 even under autocast: the layer is small and top-k selection is sensitive to precision.
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return F.linear(x.float(), self.gate.weight.float())

    def forward(self, x: torch.Tensor, causal: bool = False) -> Routing:
        """x: (T, d) flattened tokens -> Routing for this batch.

        causal=True requires each token's experts to depend on that token alone, as a decoder needs
        at evaluation and generation time. Token-choice routers always route that way.
        """
        raise NotImplementedError
