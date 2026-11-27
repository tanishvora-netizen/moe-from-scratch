"""Read the token files prepare_dataset writes and cut random (input, target) batches from them."""
import json
from pathlib import Path

import numpy as np
import torch

# Seeds the validation windows apart from --seed, so every run and checkpoint is measured on the same text.
VALIDATION_SEED = 0


def load_meta(data_dir: str | Path) -> dict:
    """meta.json: dataset, tokenizer, vocab_size, the token dtype and the train and val token counts."""
    return json.loads((Path(data_dir) / "meta.json").read_text())


def load_split(data_dir: str | Path, split: str) -> np.memmap:
    """The "train" or "val" tokens as a read-only memory map, so a large split never has to fit in RAM."""
    meta = load_meta(data_dir)
    return np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.dtype(meta["dtype"]), mode="r")


def sample_offsets(rng: np.random.Generator, data: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
    """batch_size random window starts, each leaving room for seq_len inputs plus the shifted target."""
    return rng.integers(0, len(data) - seq_len - 1, size=batch_size)


def get_batch(data: np.ndarray, offsets, seq_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns x and y, each (len(offsets), seq_len) int64 token ids, with y shifted one token ahead of x."""
    x = torch.from_numpy(np.stack([data[i:i + seq_len] for i in offsets]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + seq_len] for i in offsets]).astype(np.int64))
    if str(device).startswith("cuda"):
        # Pinned host memory lets the copy to the GPU run asynchronously.
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
