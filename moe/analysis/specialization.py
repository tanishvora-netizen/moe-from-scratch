"""The expert specialization numbers behind analyze_experts.py, which measure the tokens each routed expert handles.
joint[v, e] counts how often token id v was routed to expert e in one MoE layer.
Shapes: V the vocabulary size, E routed experts, C token categories.
"""
import string
from collections.abc import Callable
from contextlib import AbstractContextManager

import numpy as np
import torch

from ..data.batches import VALIDATION_SEED, get_batch, sample_offsets
from ..data.tokenizers import ByteTokenizer, GPT2Tokenizer
from ..model.transformer import MoETransformer


def make_categorizer(
    tokenizer_name: str, tok: ByteTokenizer | GPT2Tokenizer
) -> tuple[Callable[[int], str], list[str]]:
    """A function from token id to category name, and every category name in heatmap column order."""
    if tokenizer_name == "byte":
        def category(token_id):
            if token_id == 32:
                return "space"
            if token_id == 10:
                return "newline"
            if token_id >= 128:
                return "non-ascii"
            char = chr(token_id)
            if char in "aeiou":
                return "vowel"
            if char.islower():
                return "consonant"
            if char.isupper():
                return "upper"
            if char.isdigit():
                return "digit"
            if char in string.punctuation:
                return "punct"
            return "other"
        categories = ["space", "newline", "vowel", "consonant", "upper", "digit", "punct", "non-ascii", "other"]
    else:
        def category(token_id):
            text = tok.decode([token_id])
            core = text.strip()
            if not core:
                return "whitespace"
            if core[0] in string.punctuation:
                return "punct"
            if core[0].isdigit():
                return "digit"
            if text.startswith(" "):
                return "word-start-cap" if core[0].isupper() else "word-start"
            return "word-piece"
        categories = ["whitespace", "punct", "digit", "word-start", "word-start-cap", "word-piece"]
    return category, categories


@torch.no_grad()
def collect_joint_counts(
    model: MoETransformer,
    val_data: np.ndarray,
    n_batches: int,
    batch_size: int,
    device: str,
    ctx: AbstractContextManager,
) -> list[np.ndarray]:
    """Routes n_batches random validation batches and returns one (V, E) joint count per MoE layer."""
    cfg = model.cfg
    V, E = cfg.vocab_size, cfg.n_routed_experts
    joints = [np.zeros((V, E), dtype=np.int64) for _ in model.moe_layers()]
    rng = np.random.default_rng(VALIDATION_SEED)
    for _ in range(n_batches):
        x, _ = get_batch(val_data, sample_offsets(rng, val_data, batch_size, cfg.seq_len), cfg.seq_len, device)
        with ctx:
            _, _, info = model(x)
        token_ids = x.reshape(-1)
        for joint, routing in zip(joints, info["routing"], strict=True):
            pairs = token_ids[routing.sorted_token] * E + routing.sorted_expert   # flat index into joint
            joint += torch.bincount(pairs, minlength=V * E).view(V, E).cpu().numpy()
    return joints


def expert_load(joint: np.ndarray) -> np.ndarray:
    """(E,) share of all assignments that went to each expert."""
    return joint.sum(0) / joint.sum()


def normalized_mi(joint: np.ndarray) -> float:
    """I(token; expert) / H(expert). It is 0 when routing ignores the token and 1 when the token id decides the expert.

    p is the joint distribution, and px and pe are its token and expert marginals.
    """
    p = joint / joint.sum()
    px, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nonzero = p > 0
    mi = (p[nonzero] * np.log(p[nonzero] / (px @ pe)[nonzero])).sum()
    pe = pe[pe > 0]
    h_e = -(pe * np.log(pe)).sum()
    return float(mi / h_e) if h_e > 0 else 0.0


def layer_summary(layer: int, joint: np.ndarray) -> dict:
    """One row of summary.json: {"layer", "maxvio", "normalized_mi"}."""
    load = expert_load(joint)
    maxvio = float((load.max() - load.mean()) / load.mean())
    return {"layer": layer, "maxvio": maxvio, "normalized_mi": normalized_mi(joint.astype(np.float64))}


def layer_report(
    summary: dict, joint: np.ndarray, tok: ByteTokenizer | GPT2Tokenizer, top: int, min_count: int
) -> list[str]:
    """report.txt lines for one layer: a header, then each expert's top tokens by lift = P(token | expert) / P(token).

    summary is the layer's layer_summary row. Tokens an expert saw fewer than min_count times are left out.
    """
    load = expert_load(joint)
    p_token = joint.sum(1) / joint.sum()
    lines = [f"\n=== layer {summary['layer']} | MaxVio {summary['maxvio']:.3f} "
             f"| normalized MI {summary['normalized_mi']:.3f} ==="]
    for expert in range(joint.shape[1]):
        column = joint[:, expert]
        if column.sum() == 0:
            lines.append(f"  expert {expert:2d}: (never used)")
            continue
        lift = (column / column.sum()) / np.maximum(p_token, 1e-12)
        lift[column < min_count] = 0
        top_ids = np.argsort(-lift)[:top]
        tokens = ", ".join(f"{tok.decode([i])!r}x{lift[i]:.1f}" for i in top_ids if lift[i] > 0)
        lines.append(f"  expert {expert:2d} (load {load[expert]:.1%}): {tokens}")
    return lines


def category_lifts(
    joints: list[np.ndarray], tokenizer_name: str, tok: ByteTokenizer | GPT2Tokenizer
) -> tuple[list[np.ndarray], list[str]]:
    """Returns one (E, C) array per layer and the C category names.

    Each array holds lift = P(category | expert) / P(category), with nan for categories never seen.
    """
    category, categories = make_categorizer(tokenizer_name, tok)
    token_category = np.array([categories.index(category(i)) for i in range(joints[0].shape[0])])   # (V,)
    lifts = []
    for joint in joints:
        counts = np.zeros((joint.shape[1], len(categories)))   # (E, C)
        for index in range(len(categories)):
            counts[:, index] = joint[token_category == index].sum(0)
        p_category_given_expert = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        p_category = counts.sum(0) / counts.sum()
        lifts.append(np.where(p_category > 0, p_category_given_expert / np.maximum(p_category, 1e-12), np.nan))
    return lifts, categories
