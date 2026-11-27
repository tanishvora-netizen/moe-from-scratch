"""Checkpoints load back into an identical model, and bad configs raise ValueError."""
import pytest
import torch

from moe import ModelConfig, MoETransformer, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("router", ["aux_loss", "aux_free", "expert_choice"])
def test_checkpoint_round_trip(small_cfg, tmp_path, router):
    """The loaded model gives the same logits, and the config and training flags come back unchanged."""
    torch.manual_seed(0)
    cfg = small_cfg(router=router, moe_impl="naive")
    model = MoETransformer(cfg).eval()
    save_checkpoint(tmp_path / "ckpt.pt", model, cfg, {"tokenizer": "byte"}, step=7)

    loaded, loaded_cfg, args = load_checkpoint(tmp_path / "ckpt.pt", impl="naive")
    x = torch.randint(0, 256, (2, cfg.seq_len))
    with torch.no_grad():
        torch.testing.assert_close(loaded.eval()(x)[0], model(x)[0])
    assert loaded_cfg == cfg and args == {"tokenizer": "byte"}


def test_checkpoint_keeps_aux_free_bias(small_cfg, tmp_path):
    """The aux-free bias is a buffer, not a parameter, and the checkpoint still saves it."""
    torch.manual_seed(0)
    cfg = small_cfg(router="aux_free", moe_impl="naive")
    model = MoETransformer(cfg)
    for layer in model.moe_layers():
        layer.router.expert_bias.uniform_(-0.1, 0.1)
    save_checkpoint(tmp_path / "ckpt.pt", model, cfg, {}, step=0)

    loaded, _, _ = load_checkpoint(tmp_path / "ckpt.pt", impl="naive")
    for saved, restored in zip(model.moe_layers(), loaded.moe_layers(), strict=True):
        torch.testing.assert_close(restored.router.expert_bias, saved.router.expert_bias)


@pytest.mark.parametrize("bad", [
    pytest.param(dict(router="top1"), id="unknown-router"),
    pytest.param(dict(ffn_type="moe2"), id="unknown-ffn-type"),
    pytest.param(dict(moe_impl="cuda"), id="unknown-moe-impl"),
    pytest.param(dict(d_model=100, n_heads=8), id="d-model-not-divisible-by-heads"),
    pytest.param(dict(d_model=72, n_heads=8), id="odd-head-dim"),
    pytest.param(dict(top_k=0), id="top-k-zero"),
    pytest.param(dict(top_k=17), id="top-k-above-n-experts"),
])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ModelConfig(**bad)
