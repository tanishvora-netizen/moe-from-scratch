#!/usr/bin/env bash
# ~1 minute sanity check. Works on CPU/Mac (naive impl) or GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
IMPL=$(python -c "import torch; print('triton' if torch.cuda.is_available() else 'naive')")
python train.py --name smoke --dataset shakespeare --steps 200 --batch_size 16 \
  --eval_interval 100 --eval_iters 10 --log_interval 20 --impl "$IMPL"
