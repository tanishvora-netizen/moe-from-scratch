#!/usr/bin/env bash
# A quick check: the test suite, then a 200-step run on Shakespeare. Without a GPU it uses the naive path.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
IMPL=$(python -c "import torch; print('triton' if torch.cuda.is_available() else 'naive')")
python scripts/train.py --name smoke --dataset shakespeare --steps 200 --batch_size 16 \
  --eval_interval 100 --eval_iters 10 --log_interval 20 --impl "$IMPL"
