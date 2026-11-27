"""MoE layer: Triton path == naive path for every router; full model trains; param budget; no Triton needed."""
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from moe import ModelConfig, MoETransformer
from moe.experts.layer import MoELayer
from moe.routing import build_router

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("router", ["aux_loss", "aux_free", "expert_choice"])
def test_triton_matches_naive(small_cfg, router):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    cfg = small_cfg(router=router, moe_impl="triton")
    m_t = MoELayer(cfg).to(DEVICE)
    m_t.reset_parameters(0.05, 0.05)
    m_n = MoELayer(replace(cfg, moe_impl="naive")).to(DEVICE)
    m_n.load_state_dict(m_t.state_dict())

    x0 = torch.randn(2, 48, cfg.d_model, device=DEVICE)
    results = []
    for m in (m_t, m_n):
        x = x0.clone().requires_grad_()
        y, routing = m(x)
        (y.pow(2).sum() + routing.aux_loss).backward()
        results.append((y.detach(), x.grad, m.w_gate_up.grad, m.w_down.grad, m.router.gate.weight.grad))
    tol = dict(atol=2e-2, rtol=2e-2) if DEVICE == "cuda" else dict(atol=1e-4, rtol=1e-4)
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, **tol)


@pytest.mark.parametrize("router", ["aux_loss", "aux_free", "expert_choice"])
def test_model_overfits_one_batch(small_cfg, router):
    torch.manual_seed(0)
    cfg = small_cfg(router=router, moe_impl="naive")
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


def test_aux_free_bias_moves_toward_underloaded_experts(small_cfg):
    router = build_router(small_cfg(router="aux_free"))
    router.load_accum.copy_(torch.tensor([100., 0, 0, 0, 0, 0, 0, 0]))
    router.update_bias()
    assert router.expert_bias[0] < 0 and torch.all(router.expert_bias[1:] > 0)


def test_param_count_in_target_range():
    total, _ = MoETransformer(ModelConfig()).num_params()
    assert 6e6 <= total <= 10e6, total


def test_dense_baseline_matches_moe_active_params():
    _, active = MoETransformer(ModelConfig()).num_params()
    dense_total, _ = MoETransformer(ModelConfig(ffn_type="dense")).num_params()
    assert abs(active - dense_total) / dense_total < 0.02


def test_naive_model_runs_without_triton():
    """Machines without Triton (a Mac, a CPU-only box) must still import moe and run moe_impl="naive"."""
    code = (
        "import sys\n"
        "sys.modules['triton'] = None\n"   # every `import triton` now raises ImportError
        "import torch\n"
        "from moe import ModelConfig, MoETransformer\n"
        "model = MoETransformer(ModelConfig(moe_impl='naive'))\n"
        "logits, _, _ = model(torch.zeros(1, 8, dtype=torch.long))\n"
        "assert logits.shape == (1, 8, 256)\n"
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env={**os.environ, "PYTHONPATH": str(root)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
