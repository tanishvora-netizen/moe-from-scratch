"""Compare runs for the load-balancing ablation.

  python plot_runs.py runs/dense runs/moe_aux_loss runs/moe_aux_free runs/moe_expert_choice
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(run_dir):
    recs = [json.loads(l) for l in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in recs if "val_loss" in r], [r for r in recs if "train_loss" in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/comparison.png")
    args = ap.parse_args()

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.5))
    print(f"{'run':<22} | {'final val':>9} | {'causal val':>10} | {'MaxVio (last 10%)':>17} | {'tok/s':>8}")
    for run in args.runs:
        name = Path(run).name
        evals, train = load(run)
        line, = a1.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], label=name)
        if "val_loss_token_choice" in evals[0]:
            a1.plot([r["step"] for r in evals], [r["val_loss_token_choice"] for r in evals],
                    ls="--", color=line.get_color(), label=f"{name} (causal eval)")
        if "maxvio" in train[0]:
            a2.plot([r["step"] for r in train], [r["maxvio"] for r in train], label=name, alpha=0.8)
        tail = train[-max(1, len(train) // 10):]
        mv = sum(r.get("maxvio", 0) for r in tail) / len(tail) if "maxvio" in tail[0] else float("nan")
        tps = sorted(r["tok_per_s"] for r in train)[len(train) // 2]
        a3.bar(name, tps / 1e3)
        causal = evals[-1].get("val_loss_token_choice", float("nan"))
        print(f"{name:<22} | {evals[-1]['val_loss']:>9.4f} | {causal:>10.4f} | {mv:>17.3f} | {tps:>8.0f}")

    a1.set(xlabel="step", ylabel="val loss (nats/token)", title="Validation loss")
    a2.set(xlabel="step", ylabel="MaxVio", title="Load imbalance (0 = perfect)", yscale="symlog")
    a3.set(ylabel="k tokens / s (median)", title="Training throughput")
    a3.tick_params(axis="x", rotation=20)
    for a in (a1, a2):
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
