"""
Train a small DeepSeek-style MoE transformer (or its matched dense baseline).

  python scripts/train.py --dataset shakespeare --steps 200 --impl naive          # quick CPU/Mac smoke test
  python scripts/train.py --name aux_free --router aux_free                       # real run on a GPU
  python scripts/train.py --name dense --ffn dense                                # matched dense baseline
"""
from moe.training.options import parse_args
from moe.training.trainer import train

if __name__ == "__main__":
    train(parse_args(description=__doc__))
