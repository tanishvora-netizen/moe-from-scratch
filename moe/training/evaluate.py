"""Validation loss on fixed windows, and the eval record that train() logs and prints."""
import argparse
from contextlib import AbstractContextManager

import numpy as np
import torch

from ..data.batches import get_batch
from ..model.transformer import MoETransformer


@torch.no_grad()
def evaluate(
    model: MoETransformer,
    val_data: np.ndarray,
    val_offsets: list[np.ndarray],
    seq_len: int,
    device: str,
    ctx: AbstractContextManager,
    causal_routing: bool = False,
) -> float:
    """Mean cross-entropy on fixed validation windows, so numbers are comparable across runs.

    val_offsets holds one array of window starts per batch. ctx is the autocast context.
    """
    model.eval()
    losses = []
    for offsets in val_offsets:
        x, y = get_batch(val_data, offsets, seq_len, device)
        with ctx:
            _, _, info = model(x, y, causal_routing=causal_routing)
        losses.append(info["ce"].item())
    model.train()
    return float(np.mean(losses))


def eval_record(
    model: MoETransformer,
    args: argparse.Namespace,
    val_data: np.ndarray,
    val_offsets: list[np.ndarray],
    device: str,
    ctx: AbstractContextManager,
    step: int,
) -> dict:
    """{"step", "val_loss"}, plus "val_loss_token_choice" for expert-choice runs."""
    record = {"step": step, "val_loss": evaluate(model, val_data, val_offsets, args.seq_len, device, ctx)}
    if args.ffn == "moe" and args.router == "expert_choice":
        # Expert choice peeks at other tokens (incl. future ones) when routing, so its val loss
        # is optimistic. Also report a causal eval that routes each token on its own (top-k).
        record["val_loss_token_choice"] = evaluate(model, val_data, val_offsets, args.seq_len, device, ctx,
                                                   causal_routing=True)
    return record
