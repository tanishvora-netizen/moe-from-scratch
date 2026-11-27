"""Routing metrics logged during training: expert load, MaxVio and the fraction of dropped tokens."""
import torch

from ..routing.assignments import Routing


@torch.no_grad()
def routing_stats(routings: list[Routing]) -> dict:
    """Load statistics for one forward pass, from each MoE layer's Routing. Empty for a dense model.

    maxvio is (max_load - mean_load) / mean_load (Wang et al. 2024), averaged over layers; 0 is perfect balance.
    dropped_frac, the share of tokens no expert picked, appears only for routings that can drop tokens.
    """
    if not routings:
        return {}
    maxvio, loads, dropped = [], [], []
    for routing in routings:
        counts = routing.counts.float()
        maxvio.append(((counts.max() - counts.mean()) / counts.mean()).item())
        loads.append([round(load, 4) for load in (counts / counts.sum()).tolist()])
        if routing.can_drop_tokens:
            covered = torch.bincount(routing.sorted_token, minlength=routing.n_tokens)
            dropped.append(float((covered == 0).float().mean()))
    stats = {"maxvio": sum(maxvio) / len(maxvio), "maxvio_per_layer": maxvio, "loads": loads}
    if dropped:
        stats["dropped_frac"] = sum(dropped) / len(dropped)
    return stats
