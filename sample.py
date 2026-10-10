"""Generate text from a trained checkpoint.

  python sample.py --ckpt runs/moe_aux_free/ckpt.pt --prompt "Once upon a time"
"""
import argparse
from contextlib import nullcontext

import torch

from data import get_tokenizer
from moe.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--num_samples", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, run_args = load_checkpoint(args.ckpt, device, impl="triton" if device == "cuda" else "naive")
    model.eval()
    model.set_router_eval_mode("token_choice")   # causal routing for expert-choice models
    tok = get_tokenizer(run_args.get("tokenizer", "byte"))
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    idx = torch.tensor([tok.encode(args.prompt)] * args.num_samples, device=device)
    with ctx:
        out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    for i in range(args.num_samples):
        print(f"----- sample {i} -----\n{tok.decode(out[i].tolist())}\n")


if __name__ == "__main__":
    main()
