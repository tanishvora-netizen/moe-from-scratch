"""MoE layer: Triton path == naive path for every router; full model trains; param budget."""
from dataclasses import replace

import pytest
import torch

from moe.config import ModelConfig
from moe.model import MoETransformer
from moe.moe_layer import MoE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMALL = dict(d_model=64, n_heads=4, n_layers=2, seq_len=32, n_routed_experts=8, expert_hidden=32,
             top_k=2, n_shared_experts=1, vocab_size=256)


@pytest.mark.parametrize("router", ["aux_loss", "aux_free", "expert_choice"])
def test_triton_matches_naive(router):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    cfg = ModelConfig(**SMALL, router=router, moe_impl="triton")
    m_t = MoE(cfg).to(DEVICE)
    m_t.reset_parameters(0.05, 0.05)
    m_n = MoE(replace(cfg, moe_impl="naive")).to(DEVICE)
    m_n.load_state_dict(m_t.state_dict())

    x0 = torch.randn(2, 48, cfg.d_model, device=DEVICE)
    results = []
    for m in (m_t, m_n):
        x = x0.clone().requires_grad_()
        y, aux = m(x)
        (y.pow(2).sum() + aux).backward()
        results.append((y.detach(), x.grad, m.w_gate_up.grad, m.w_down.grad, m.router.gate.weight.grad))
    tol = dict(atol=2e-2, rtol=2e-2) if DEVICE == "cuda" else dict(atol=1e-4, rtol=1e-4)
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, **tol)


@pytest.mark.parametrize("router", ["aux_loss", "aux_free", "expert_choice"])
def test_model_overfits_one_batch(router):
    torch.manual_seed(0)
    cfg = ModelConfig(**SMALL, router=router, moe_impl="naive")
    model = MoETransformer(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 256, (4, 33), device=DEVICE)
    first = None
    for _ in range(80):
        _, loss, info = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_router_biases()
        first = first or info["ce"].item()
    assert info["ce"].item() < 0.5 * first


def test_aux_free_bias_moves_toward_underloaded_experts():
    m = MoE(ModelConfig(**SMALL, router="aux_free", moe_impl="naive"))
    m.router.load_accum.copy_(torch.tensor([100., 0, 0, 0, 0, 0, 0, 0]))
    m.router.update_bias()
    assert m.router.expert_bias[0] < 0 and torch.all(m.router.expert_bias[1:] > 0)


def test_param_count_in_target_range():
    total, active = MoETransformer(ModelConfig()).num_params()
    assert 6e6 <= total <= 10e6, total
    dense_total, _ = MoETransformer(ModelConfig(ffn_type="dense")).num_params()
    assert abs(active - dense_total) / dense_total < 0.02   # dense baseline matches MoE active params
