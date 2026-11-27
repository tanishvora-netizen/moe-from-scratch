"""Compares the runs of the load-balancing ablation.

    python scripts/plot_runs.py runs/dense runs/moe_aux_loss runs/moe_aux_free runs/moe_expert_choice

It saves a 3-panel plot and prints a Markdown table that can be pasted into the README as is.
"""
import argparse

from moe.analysis.plots import plot_run_comparison
from moe.analysis.runs import load_run, results_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="runs/comparison.png")
    args = parser.parse_args()

    runs = [load_run(run_dir) for run_dir in args.runs]
    print(results_table(runs))
    plot_run_comparison(runs, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
