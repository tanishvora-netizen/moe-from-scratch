"""The Router base class, which holds the gate projection and computes the fp32 logits every router uses.
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
        """Returns (T, E) fp32 scores for x of shape (T, d)."""
        # fp32 even under autocast, because top-k selection is sensitive to precision and the layer is small.
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return F.linear(x.float(), self.gate.weight.float())

    def forward(self, x: torch.Tensor, causal: bool = False) -> Routing:
        """Routes x, the (T, d) flattened tokens of one batch.

        causal=True requires each token's experts to depend on that token alone, as a decoder needs
        at evaluation and generation time. Token-choice routers always route that way.
        """
        raise NotImplementedError
