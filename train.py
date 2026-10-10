"""
Train a small DeepSeek-style MoE transformer (or its matched dense baseline).

  python train.py --dataset shakespeare --steps 200 --impl naive          # quick CPU/Mac smoke test
  python train.py --name aux_free --router aux_free                       # real run on a GPU
  python train.py --name dense --ffn dense                                # matched dense baseline
"""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data import get_batch, get_tokenizer, load_meta, load_split, prepare_dataset, sample_offsets
from moe.config import ModelConfig
from moe.model import MoETransformer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # run
    p.add_argument("--name", default=None, help="run name (default: derived from ffn/router)")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    # data
    p.add_argument("--dataset", default="tinystories", choices=["shakespeare", "tinystories"])
    p.add_argument("--tokenizer", default="byte", choices=["byte", "gpt2"])
    p.add_argument("--max_mb", type=int, default=200, help="MB of TinyStories text to use")
    # model
    p.add_argument("--ffn", default="moe", choices=["moe", "dense"])
    p.add_argument("--router", default="aux_loss", choices=["aux_loss", "aux_free", "expert_choice"])
    p.add_argument("--impl", default="triton", choices=["triton", "naive"])
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_experts", type=int, default=16)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--expert_hidden", type=int, default=64)
    p.add_argument("--n_shared", type=int, default=2)
    p.add_argument("--first_k_dense", type=int, default=0)
    p.add_argument("--dense_hidden", type=int, default=None, help="default: matched to MoE active params")
    p.add_argument("--aux_loss_coef", type=float, default=0.01)
    p.add_argument("--bias_update_rate", type=float, default=1e-3)
    p.add_argument("--capacity_factor", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    # optimization
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=25)
    return p.parse_args()


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(arg, device):
    if device != "cuda":
        return torch.float32
    if arg == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[arg]


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model, val_data, val_offsets, seq_len, device, ctx):
    model.eval()
    losses = []
    for ix in val_offsets:                 # fixed windows -> numbers comparable across runs
        x, y = get_batch(val_data, ix, seq_len, device)
        with ctx:
            _, _, info = model(x, y)
        losses.append(info["ce"].item())
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    device = pick_device(args.device)
    if args.impl == "triton" and device != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        print("[warn] Triton kernels need an NVIDIA GPU -> using --impl naive")
        args.impl = "naive"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dtype = pick_dtype(args.dtype, device)
    ctx = torch.autocast("cuda", dtype=dtype) if (device == "cuda" and dtype != torch.float32) else nullcontext()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- data ----
    data_dir = prepare_dataset(args.dataset, args.tokenizer, args.max_mb)
    meta = load_meta(data_dir)
    train_data, val_data = load_split(data_dir, "train"), load_split(data_dir, "val")

    # ---- model ----
    cfg = ModelConfig(
        vocab_size=meta["vocab_size"], seq_len=args.seq_len, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, ffn_type=args.ffn,
        n_routed_experts=args.n_experts, expert_hidden=args.expert_hidden, top_k=args.top_k,
        n_shared_experts=args.n_shared, first_k_dense=args.first_k_dense, router=args.router,
        aux_loss_coef=args.aux_loss_coef, bias_update_rate=args.bias_update_rate,
        capacity_factor=args.capacity_factor, moe_impl=args.impl,
    )
    cfg.dense_hidden = args.dense_hidden or cfg.matched_dense_hidden
    model = MoETransformer(cfg).to(device)
    total, active = model.num_params()
    is_moe = args.ffn == "moe"

    name = args.name or (f"moe_{args.router}" if is_moe else "dense")
    run_dir = Path(args.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"model": asdict(cfg), "args": vars(args)}, indent=2))
    print(f"[run] {name} | device={device} dtype={dtype} impl={args.impl if is_moe else '-'}")
    print(f"[run] params: total={total / 1e6:.2f}M  active/token={active / 1e6:.2f}M")

    opt = model.configure_optimizer(args.weight_decay, args.lr, device)
    scaler = make_scaler(dtype == torch.float16)
    log_f = open(run_dir / "metrics.jsonl", "w")

    def log(rec):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    val_rng = np.random.default_rng(0)
    val_offsets = [sample_offsets(val_rng, val_data, args.batch_size, args.seq_len) for _ in range(args.eval_iters)]
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len

    t0, steps_since = time.time(), 0
    for step in range(args.steps + 1):
        # ---- eval + checkpoint ----
        if step % args.eval_interval == 0 or step == args.steps:
            rec = {"step": step, "val_loss": evaluate(model, val_data, val_offsets, args.seq_len, device, ctx)}
            msg = f"[eval] step {step:5d} | val_loss {rec['val_loss']:.4f}"
            if is_moe and args.router == "expert_choice":
                # Expert choice peeks at other tokens (incl. future ones) when routing. Also
                # report a causal eval that routes each token on its own (top-k per token).
                model.set_router_eval_mode("token_choice")
                rec["val_loss_token_choice"] = evaluate(model, val_data, val_offsets, args.seq_len, device, ctx)
                model.set_router_eval_mode(None)
                msg += f" | causal (token-choice) val_loss {rec['val_loss_token_choice']:.4f}"
            print(msg)
            log(rec)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "args": vars(args), "step": step},
                       run_dir / "ckpt.pt")
            t0, steps_since = time.time(), 0
        if step == args.steps:
            break

        # ---- train step ----
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, sample_offsets(rng, train_data, args.batch_size, args.seq_len),
                             args.seq_len, device)
            with ctx:
                _, loss, info = model(x, y)
            scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        model.update_router_biases()      # no-op unless router == aux_free
        steps_since += 1

        if step % args.log_interval == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            rec = {"step": step, "train_loss": loss.item(), "ce": info["ce"].item(), "aux": info["aux"].item(),
                   "lr": lr, "grad_norm": float(gnorm), "tok_per_s": tokens_per_step * steps_since / dt}
            rec.update(model.routing_stats())
            msg = (f"step {step:5d} | ce {rec['ce']:.4f} | lr {lr:.2e} | "
                   f"{rec['tok_per_s'] / 1e3:.0f}k tok/s")
            if "maxvio" in rec:
                msg += f" | MaxVio {rec['maxvio']:.3f}"
            if args.router == "aux_loss" and is_moe:
                msg += f" | aux {rec['aux']:.3f}"
            if "dropped_frac" in rec:
                msg += f" | dropped {rec['dropped_frac']:.1%}"
            print(msg)
            log(rec)
            t0, steps_since = time.time(), 0

    # ---- sample ----
    tok = get_tokenizer(args.tokenizer)
    model.eval()
    model.set_router_eval_mode("token_choice")   # only affects expert-choice models
    prompt = "Once upon a time" if args.dataset == "tinystories" else "ROMEO:"
    idx = torch.tensor([tok.encode(prompt)], device=device)
    with ctx:
        out = model.generate(idx, max_new_tokens=400)
    text = tok.decode(out[0].tolist())
    (run_dir / "sample.txt").write_text(text)
    print("\n----- sample -----\n" + text)
    log_f.close()


if __name__ == "__main__":
    main()
