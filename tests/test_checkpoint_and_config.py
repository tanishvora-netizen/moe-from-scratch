"""Checkpoints load back into an identical model; bad configs fail loudly."""
from dataclasses import asdict

import pytest
import torch

from moe.config import ModelConfig
from moe.model import MoETransformer, load_checkpoint


@pytest.mark.parametrize("router", ["aux_loss", "aux_free", "expert_choice"])
def test_checkpoint_round_trip(small_cfg, tmp_path, router):
    torch.manual_seed(0)
    cfg = small_cfg(router=router, moe_impl="naive")
    model = MoETransformer(cfg).eval()
    if router == "aux_free":   # the bias is a buffer, not a parameter: make sure it is saved too
        for m in model.moe_layers():
            m.router.expert_bias.uniform_(-0.1, 0.1)
    # same layout scripts/train.py writes
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "args": {"tokenizer": "byte"}, "step": 7},
               tmp_path / "ckpt.pt")

    loaded, loaded_cfg, args = load_checkpoint(tmp_path / "ckpt.pt", impl="naive")
    x = torch.randint(0, 256, (2, cfg.seq_len))
    with torch.no_grad():
        torch.testing.assert_close(loaded.eval()(x)[0], model(x)[0])
    assert loaded_cfg == cfg and args == {"tokenizer": "byte"}


@pytest.mark.parametrize("bad", [
    dict(router="top1"),
    dict(ffn_type="moe2"),
    dict(moe_impl="cuda"),
    dict(d_model=100, n_heads=8),     # not divisible
    dict(d_model=72, n_heads=8),      # head_dim 9 is odd, RoPE needs pairs
    dict(top_k=0),
    dict(top_k=17),                   # more than n_routed_experts
])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ModelConfig(**bad)
