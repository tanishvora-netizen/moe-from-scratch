#!/usr/bin/env bash
# The full experiment on one GPU: the benchmark, the dense baseline and the three routers,
# then the comparison plot and the expert analysis of each MoE run.
# Usage: STEPS=5000 bash scripts/run_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-5000}
COMMON="--dataset tinystories --steps $STEPS --seed 1337"

python scripts/benchmark.py

python scripts/train.py --name dense             --ffn dense                    $COMMON
python scripts/train.py --name moe_aux_loss      --ffn moe --router aux_loss      $COMMON
python scripts/train.py --name moe_aux_free      --ffn moe --router aux_free      $COMMON
python scripts/train.py --name moe_expert_choice --ffn moe --router expert_choice $COMMON

python scripts/plot_runs.py runs/dense runs/moe_aux_loss runs/moe_aux_free runs/moe_expert_choice \
  --out runs/comparison.png

for r in moe_aux_loss moe_aux_free moe_expert_choice; do
  python scripts/analyze_experts.py --ckpt runs/$r/ckpt.pt
done
