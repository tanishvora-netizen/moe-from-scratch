"""The specialization heatmap and load plot for analyze_experts.py, and the run comparison for plot_runs.py.
This module imports matplotlib, which analyze_experts.py treats as optional.
Shapes: E routed experts, C token categories.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .runs import Run, median_tok_per_s


def plot_specialization(
    path: Path, lifts: list[np.ndarray], categories: list[str], summary: list[dict], run_name: str
) -> None:
    """One heatmap per MoE layer of log2 lift, experts by token categories.

    lifts holds one (E, C) array per layer from category_lifts, and summary one layer_summary row per layer.
    """
    n_layers, E = len(lifts), lifts[0].shape[0]
    fig, axes = plt.subplots(1, n_layers, figsize=(3.2 * n_layers, 5), squeeze=False)
    for layer, ax in enumerate(axes[0]):
        image = ax.imshow(np.log2(np.clip(lifts[layer], 1e-3, None)), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_title(f"layer {layer}  (nMI {summary[layer]['normalized_mi']:.2f})", fontsize=9)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=70, fontsize=7)
        ax.set_yticks(range(E))
        ax.set_yticklabels([f"e{expert}" for expert in range(E)], fontsize=7)
    fig.colorbar(image, ax=axes[0].tolist(), label="log2 lift  P(cat|expert)/P(cat)", shrink=0.8)
    fig.suptitle(f"Expert specialization: {run_name}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_load(path: Path, loads: list[np.ndarray]) -> None:
    """Each layer's (E,) share of assignments per expert, against the uniform 1/E line."""
    E = len(loads[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    for layer, load in enumerate(loads):
        ax.plot(range(E), load, marker="o", label=f"layer {layer}")
    ax.axhline(1 / E, color="k", ls="--", lw=1, label="uniform")
    ax.set(xlabel="expert", ylabel="fraction of assignments", title="Expert load")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run_comparison(runs: list[Run], path: str | Path) -> None:
    """Three panels: validation loss over steps, MaxVio over steps, and median training throughput."""
    fig, (loss_ax, maxvio_ax, speed_ax) = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, evals, train in runs:
        eval_steps = [record["step"] for record in evals]
        line, = loss_ax.plot(eval_steps, [record["val_loss"] for record in evals], label=name)
        if "val_loss_token_choice" in evals[0]:
            loss_ax.plot(eval_steps, [record["val_loss_token_choice"] for record in evals],
                         ls="--", color=line.get_color(), label=f"{name} (causal eval)")
        if "maxvio" in train[0]:
            maxvio_ax.plot([record["step"] for record in train], [record["maxvio"] for record in train],
                           label=name, alpha=0.8)
        speed_ax.bar(name, median_tok_per_s(train) / 1e3)

    loss_ax.set(xlabel="step", ylabel="val loss (nats/token)", title="Validation loss")
    maxvio_ax.set(xlabel="step", ylabel="MaxVio", title="Load imbalance (0 = perfect)", yscale="symlog")
    speed_ax.set(ylabel="k tokens / s (median)", title="Training throughput")
    speed_ax.tick_params(axis="x", rotation=20)
    for ax in (loss_ax, maxvio_ax):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
