"""What training logs: routing stats (expert load, MaxVio, dropped tokens), the console lines, and metrics.jsonl."""
import json
from typing import TextIO

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


def format_eval(record: dict) -> str:
    """The console line for an eval record."""
    line = f"[eval] step {record['step']:5d} | val_loss {record['val_loss']:.4f}"
    if "val_loss_token_choice" in record:
        line += f" | causal (token-choice) val_loss {record['val_loss_token_choice']:.4f}"
    return line


def format_train(record: dict) -> str:
    """The console line for a train record. Routing stats appear only when the run has them."""
    line = (f"step {record['step']:5d} | ce {record['ce']:.4f} | lr {record['lr']:.2e} | "
            f"{record['tok_per_s'] / 1e3:.0f}k tok/s")
    if "maxvio" in record:
        line += f" | MaxVio {record['maxvio']:.3f}"
    if record["aux"] > 0:
        line += f" | aux {record['aux']:.3f}"
    if "dropped_frac" in record:
        line += f" | dropped {record['dropped_frac']:.1%}"
    return line


def write_jsonl(file: TextIO, record: dict) -> None:
    """Writes record as one JSON line and flushes, so the log is complete up to a crash and can be followed live."""
    file.write(json.dumps(record) + "\n")
    file.flush()
