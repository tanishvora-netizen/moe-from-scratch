"""
Expert-specialization analysis: which tokens does each expert handle?

  python scripts/analyze_experts.py --ckpt runs/moe_aux_free/ckpt.pt

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
from pathlib import Path

import torch

from moe import load_checkpoint
from moe.analysis.specialization import category_lifts, collect_joint_counts, expert_load, layer_report, layer_summary
from moe.data.batches import load_split
from moe.data.datasets import dataset_dir
from moe.data.tokenizers import get_tokenizer
from moe.training.runtime import autocast_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top", type=int, default=8, help="top tokens to list per expert")
    parser.add_argument("--min_count", type=int, default=30, help="min occurrences for a token to be listed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, run_args = load_checkpoint(args.ckpt, device, impl="triton" if device == "cuda" else "naive")
    assert cfg.ffn_type == "moe", "this checkpoint is a dense model"
    model.eval()
    tokenizer_name = run_args.get("tokenizer", "byte")
    tokenizer = get_tokenizer(tokenizer_name)
    data_dir = run_args.get("data_dir") or dataset_dir(
        run_args.get("dataset", "tinystories"), tokenizer_name, run_args.get("max_mb", 200))
    joints = collect_joint_counts(model, load_split(data_dir, "val"), args.batches, args.batch_size, device,
                                  autocast_context(device, torch.bfloat16))

    summary = [layer_summary(layer, joint) for layer, joint in enumerate(joints)]
    report = []
    for row, joint in zip(summary, joints, strict=True):
        report += layer_report(row, joint, tokenizer, args.top, args.min_count)

    out_dir = Path(args.ckpt).parent / "analysis"
    out_dir.mkdir(exist_ok=True)
    print(f"{'layer':>5} | {'MaxVio':>7} | {'norm MI':>7}")
    for row in summary:
        print(f"{row['layer']:>5} | {row['maxvio']:>7.3f} | {row['normalized_mi']:>7.3f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.txt").write_text("\n".join(report))
    print("\n".join(report[: cfg.n_routed_experts + 1]))   # layer 0 preview
    print(f"\nfull report: {out_dir / 'report.txt'}")

    try:
        from moe.analysis.plots import plot_load, plot_specialization
    except ImportError:   # matplotlib is optional
        return
    lifts, categories = category_lifts(joints, tokenizer_name, tokenizer)
    plot_specialization(out_dir / "specialization.png", lifts, categories, summary, Path(args.ckpt).parent.name)
    plot_load(out_dir / "load.png", [expert_load(joint) for joint in joints])
    print(f"plots: {out_dir / 'specialization.png'}, {out_dir / 'load.png'}")


if __name__ == "__main__":
    main()
