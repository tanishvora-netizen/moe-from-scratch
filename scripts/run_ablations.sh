#!/usr/bin/env bash
# Full experiment on one GPU: dense baseline + 3 load-balancing strategies,
# then the comparison plot and expert-specialization analysis.
#   STEPS=5000 bash scripts/run_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-5000}
COMMON="--dataset tinystories --steps $STEPS --seed 1337"

python benchmark.py

python train.py --name dense             --ffn dense                    $COMMON
python train.py --name moe_aux_loss      --ffn moe --router aux_loss      $COMMON
python train.py --name moe_aux_free      --ffn moe --router aux_free      $COMMON
python train.py --name moe_expert_choice --ffn moe --router expert_choice $COMMON

python plot_runs.py runs/dense runs/moe_aux_loss runs/moe_aux_free runs/moe_expert_choice \
  --out runs/comparison.png

for r in moe_aux_loss moe_aux_free moe_expert_choice; do
  python analyze_experts.py --ckpt runs/$r/ckpt.pt
done
