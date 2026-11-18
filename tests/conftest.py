"""Shared test setup.

Without a GPU, the Triton kernels run in Triton's CPU interpreter. TRITON_INTERPRET has to be set
before triton is imported, which is why it happens here at collection time.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests run without `pip install -e .` too

try:
    import torch
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
except ImportError:
    pass

from moe.config import ModelConfig  # noqa: E402

# Small enough that every test runs in seconds on a CPU, big enough to have several experts per token.
SMALL = dict(d_model=64, n_heads=4, n_layers=2, seq_len=32, n_routed_experts=8, expert_hidden=32,
             top_k=2, n_shared_experts=1, vocab_size=256)


@pytest.fixture
def small_cfg():
    """small_cfg(router="aux_free", ...) -> a tiny ModelConfig with the given overrides."""
    def make(**overrides):
        return ModelConfig(**{**SMALL, **overrides})
    return make
