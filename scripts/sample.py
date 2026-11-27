"""Generate text from a trained checkpoint.

  python scripts/sample.py --ckpt runs/moe_aux_free/ckpt.pt --prompt "Once upon a time"
"""
import argparse

import torch

from moe import load_checkpoint
from moe.data.tokenizers import get_tokenizer
from moe.training.runtime import autocast_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, run_args = load_checkpoint(args.ckpt, device, impl="triton" if device == "cuda" else "naive")
    model.eval()
    tokenizer = get_tokenizer(run_args.get("tokenizer", "byte"))

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)] * args.num_samples, device=device)
    with autocast_context(device, torch.bfloat16):
        out = model.generate(prompt_ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
    for i in range(args.num_samples):
        print(f"----- sample {i} -----\n{tokenizer.decode(out[i].tolist())}\n")


if __name__ == "__main__":
    main()
