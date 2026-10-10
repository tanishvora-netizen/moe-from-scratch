"""
Expert-specialization analysis: which tokens does each expert handle?

  python analyze_experts.py --ckpt runs/moe_aux_free/ckpt.pt

Outputs (in <run>/analysis/):
  specialization.png  per-layer heatmap of lift = P(category | expert) / P(category)
  load.png            per-layer expert load
  report.txt          top tokens per expert (by lift) for every layer
  summary.json        per-layer MaxVio and normalized mutual information I(token; expert) / H(expert)

Normalized MI is a single "how specialized are the experts" number you can compare
across routing strategies: 0 = routing ignores token identity, 1 = token id fully
determines the expert.
"""
import argparse
import json
import string
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from data import DATA_ROOT, get_batch, get_tokenizer, load_split, sample_offsets
from moe.model import load_checkpoint


def make_categorizer(tokenizer_name, tok):
    if tokenizer_name == "byte":
        def cat(i):
            if i == 32:
                return "space"
            if i == 10:
                return "newline"
            if i >= 128:
                return "non-ascii"
            c = chr(i)
            if c in "aeiou":
                return "vowel"
            if c.islower():
                return "consonant"
            if c.isupper():
                return "upper"
            if c.isdigit():
                return "digit"
            if c in string.punctuation:
                return "punct"
            return "other"
        cats = ["space", "newline", "vowel", "consonant", "upper", "digit", "punct", "non-ascii", "other"]
    else:
        def cat(i):
            s = tok.decode([i])
            core = s.strip()
            if not core:
                return "whitespace"
            if core[0] in string.punctuation:
                return "punct"
            if core[0].isdigit():
                return "digit"
            if s.startswith(" "):
                return "word-start-cap" if core[0].isupper() else "word-start"
            return "word-piece"
        cats = ["whitespace", "punct", "digit", "word-start", "word-start-cap", "word-piece"]
    return cat, cats


def show(tok, i):
    return repr(tok.decode([i]))


def normalized_mi(joint):
    p = joint / joint.sum()
    px, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (px @ pe)[nz])).sum()
    pe = pe[pe > 0]
    h_e = -(pe * np.log(pe)).sum()
    return float(mi / h_e) if h_e > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--top", type=int, default=8, help="top tokens to list per expert")
    ap.add_argument("--min_count", type=int, default=30, help="min occurrences for a token to be listed")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, run_args = load_checkpoint(args.ckpt, device, impl="triton" if device == "cuda" else "naive")
    assert cfg.ffn_type == "moe", "this checkpoint is a dense model"
    model.eval()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    tokenizer_name = run_args.get("tokenizer", "byte")
    tok = get_tokenizer(tokenizer_name)
    suffix = f"_{run_args.get('max_mb', 200)}mb" if run_args.get("dataset") == "tinystories" else ""
    data_dir = DATA_ROOT / f"{run_args.get('dataset', 'tinystories')}_{tokenizer_name}{suffix}"
    val = load_split(data_dir, "val")

    layers = model.moe_layers()
    for m in layers:
        m.record = True
    V, E = cfg.vocab_size, cfg.n_routed_experts
    joint = [np.zeros((V, E), dtype=np.int64) for _ in layers]

    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(args.batches):
            x, _ = get_batch(val, sample_offsets(rng, val, args.batch_size, cfg.seq_len), cfg.seq_len, device)
            with ctx:
                model(x)
            ids = x.reshape(-1)
            for li, m in enumerate(layers):
                tok_idx, exp = m.last_assign
                flat = ids[tok_idx] * E + exp
                joint[li] += torch.bincount(flat, minlength=V * E).view(V, E).cpu().numpy()

    out_dir = Path(args.ckpt).parent / "analysis"
    out_dir.mkdir(exist_ok=True)
    cat_of, cats = make_categorizer(tokenizer_name, tok)
    tok_cat = np.array([cats.index(cat_of(i)) for i in range(V)])

    summary, lifts, loads, report = [], [], [], []
    for li, J in enumerate(joint):
        load = J.sum(0) / J.sum()
        maxvio = float((load.max() - load.mean()) / load.mean())
        nmi = normalized_mi(J.astype(np.float64))
        summary.append({"layer": li, "maxvio": maxvio, "normalized_mi": nmi})
        loads.append(load)

        # category lift matrix: P(cat | expert) / P(cat)
        C = np.zeros((E, len(cats)))
        for c in range(len(cats)):
            C[:, c] = J[tok_cat == c].sum(0)
        p_cat_given_e = C / np.maximum(C.sum(1, keepdims=True), 1)
        p_cat = C.sum(0) / C.sum()
        lifts.append(np.where(p_cat > 0, p_cat_given_e / np.maximum(p_cat, 1e-12), np.nan))

        # top tokens per expert by lift = P(token | expert) / P(token)
        p_tok = J.sum(1) / J.sum()
        report.append(f"\n=== layer {li} | MaxVio {maxvio:.3f} | normalized MI {nmi:.3f} ===")
        for e in range(E):
            col = J[:, e]
            if col.sum() == 0:
                report.append(f"  expert {e:2d}: (never used)")
                continue
            lift = (col / col.sum()) / np.maximum(p_tok, 1e-12)
            lift[col < args.min_count] = 0
            top = np.argsort(-lift)[: args.top]
            toks = ", ".join(f"{show(tok, i)}x{lift[i]:.1f}" for i in top if lift[i] > 0)
            report.append(f"  expert {e:2d} (load {load[e]:.1%}): {toks}")

    print(f"{'layer':>5} | {'MaxVio':>7} | {'norm MI':>7}")
    for s in summary:
        print(f"{s['layer']:>5} | {s['maxvio']:>7.3f} | {s['normalized_mi']:>7.3f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.txt").write_text("\n".join(report))
    print("\n".join(report[: E + 1]))   # layer 0 preview
    print(f"\nfull report: {out_dir / 'report.txt'}")

    try:
        import matplotlib.pyplot as plt
        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 5), squeeze=False)
        for li, ax in enumerate(axes[0]):
            im = ax.imshow(np.log2(np.clip(lifts[li], 1e-3, None)), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
            ax.set_title(f"layer {li}  (nMI {summary[li]['normalized_mi']:.2f})", fontsize=9)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=70, fontsize=7)
            ax.set_yticks(range(E))
            ax.set_yticklabels([f"e{e}" for e in range(E)], fontsize=7)
        fig.colorbar(im, ax=axes[0].tolist(), label="log2 lift  P(cat|expert)/P(cat)", shrink=0.8)
        fig.suptitle(f"Expert specialization: {Path(args.ckpt).parent.name}")
        fig.savefig(out_dir / "specialization.png", dpi=150, bbox_inches="tight")

        fig, ax = plt.subplots(figsize=(8, 4))
        for li, load in enumerate(loads):
            ax.plot(range(E), load, marker="o", label=f"layer {li}")
        ax.axhline(1 / E, color="k", ls="--", lw=1, label="uniform")
        ax.set(xlabel="expert", ylabel="fraction of assignments", title="Expert load")
        ax.legend(fontsize=7)
        fig.savefig(out_dir / "load.png", dpi=150, bbox_inches="tight")
        print(f"plots: {out_dir / 'specialization.png'}, {out_dir / 'load.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
