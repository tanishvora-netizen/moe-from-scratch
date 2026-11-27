"""The three routers of the load-balancing ablation, picked by ModelConfig.router.
All three return the same sorted Routing layout, so the expert computation is the same for each.
"""
from ..config import ModelConfig
from .aux_free import AuxFreeRouter
from .aux_loss import AuxLossRouter
from .base import Router
from .expert_choice import ExpertChoiceRouter

ROUTERS = {"aux_loss": AuxLossRouter, "aux_free": AuxFreeRouter, "expert_choice": ExpertChoiceRouter}


def build_router(cfg: ModelConfig) -> Router:
    return ROUTERS[cfg.router](cfg)
