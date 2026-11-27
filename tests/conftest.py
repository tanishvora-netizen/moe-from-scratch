"""Shared test setup. Without a GPU the Triton kernels run in Triton's CPU interpreter, and TRITON_INTERPRET
must be set before triton is imported, so this file sets it at collection time.
"""
import os

import pytest

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
    """small_cfg(router="aux_free", ...) returns a SMALL ModelConfig with those overrides."""
    def make(**overrides):
        return ModelConfig(**{**SMALL, **overrides})
    return make
