"""Router: the shared Routing layout, the balance-loss formula, and which routers are causal.
Shapes: T tokens, E routed experts, k experts per token, M assignments.
"""
import pytest
import torch
import torch.nn as nn

from moe import MoETransformer
from moe.routing import build_router

ROUTERS = ["aux_loss", "aux_free", "expert_choice"]


@pytest.mark.parametrize("strategy", ROUTERS)
def test_routing_layout_is_consistent(small_cfg, strategy):
    """Every router returns the Routing layout the kernels assume."""
    torch.manual_seed(0)
    cfg = small_cfg(router=strategy)
    router = build_router(cfg)
    nn.init.normal_(router.gate.weight, std=1.0)   # big logits, so routing is far from uniform
    T, E, k = 50, cfg.n_routed_experts, cfg.top_k
    r = router(torch.randn(T, cfg.d_model))
    M = r.sorted_token.numel()

    assert r.offsets[0] == 0 and r.offsets[-1] == M
    assert torch.equal(r.offsets.diff(), r.counts)
    assert torch.all(r.sorted_expert.diff() >= 0), "assignments must be grouped by expert"
    assert torch.equal(torch.bincount(r.sorted_expert, minlength=E), r.counts)
    assert (r.sorted_token * E + r.sorted_expert).unique().numel() == M, "duplicate (token, expert) pair"

    if strategy == "expert_choice":
        assert torch.all(r.counts == r.counts[0]), "expert choice should give every expert the same load"
    else:
        assert M == T * k
        assert torch.all(torch.bincount(r.sorted_token, minlength=T) == k), "every token gets exactly k experts"
        gate_sum = torch.zeros(T).index_add_(0, r.sorted_token, r.sorted_weight)
        torch.testing.assert_close(gate_sum, torch.ones(T))


def test_aux_loss_is_one_for_a_uniform_router(small_cfg):
    router = build_router(small_cfg(router="aux_loss"))
    nn.init.zeros_(router.gate.weight)             # every expert gets probability 1/E
    r = router(torch.randn(64, router.gate.in_features))
    torch.testing.assert_close(r.aux_loss, torch.tensor(1.0))


def test_aux_loss_reaches_e_over_k_when_routing_collapses(small_cfg):
    """When every token goes to the same k experts, E * sum(f * P) reaches its top-k maximum, E / k."""
    cfg = small_cfg(router="aux_loss")
    router = build_router(cfg)
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[: cfg.top_k] = 1.0     # experts 0 to k-1 win for every positive input
    r = router(torch.ones(64, cfg.d_model))
    expected = cfg.n_routed_experts / cfg.top_k
    torch.testing.assert_close(r.aux_loss, torch.tensor(expected), atol=1e-3, rtol=1e-3)


def _prefix_logits_change(model, n_prefix=16, causal_routing=False):
    """Max change in the logits of the first n_prefix positions when only later tokens change."""
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, model.cfg.seq_len))
    x_other_future = x.clone()
    x_other_future[:, n_prefix:] = torch.randint(0, 256, (2, model.cfg.seq_len - n_prefix))
    with torch.no_grad():
        a = model(x, causal_routing=causal_routing)[0][:, :n_prefix]
        b = model(x_other_future, causal_routing=causal_routing)[0][:, :n_prefix]
    return (a - b).abs().max().item()


@pytest.mark.parametrize("strategy", ["aux_loss", "aux_free"])
def test_token_choice_routers_are_causal(small_cfg, strategy):
    torch.manual_seed(0)
    model = MoETransformer(small_cfg(router=strategy, moe_impl="naive")).eval()
    assert _prefix_logits_change(model) < 1e-5


def test_expert_choice_sees_future_tokens_unless_evaluated_with_token_choice(small_cfg):
    """Expert choice lets later tokens change earlier logits, the leak in the README, unless routing is causal."""
    torch.manual_seed(0)
    model = MoETransformer(small_cfg(router="expert_choice", moe_impl="naive")).eval()
    assert _prefix_logits_change(model) > 1e-4

    assert _prefix_logits_change(model, causal_routing=True) < 1e-5
