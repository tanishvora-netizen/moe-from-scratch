"""Checkpoint files: model weights, the ModelConfig, the training flags and the step, in one ckpt.pt."""
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig
from .model.transformer import MoETransformer


def save_checkpoint(path: str | Path, model: MoETransformer, cfg: ModelConfig, args: dict, step: int) -> None:
    """args is the training flags as a plain dict, which load_checkpoint hands back."""
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "args": args, "step": step}, path)


def load_checkpoint(
    path: str | Path, device: str = "cpu", impl: str | None = None
) -> tuple[MoETransformer, ModelConfig, dict]:
    """Rebuilds the model from a ckpt.pt. Returns (model, cfg, training flags).

    impl overrides the saved moe_impl, e.g. to load a GPU-trained checkpoint with impl="naive" on a laptop.
    """
    # The checkpoint only holds tensors and plain Python values, so the safe loader works
    # and a downloaded checkpoint can't run arbitrary code through pickle.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg_dict = dict(ckpt["config"])
    if impl is not None:
        cfg_dict["moe_impl"] = impl
    cfg = ModelConfig(**cfg_dict)
    model = MoETransformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg, ckpt.get("args", {})
