"""The training loop behind scripts/train.py: build the model from the flags, train, evaluate, checkpoint, sample.
One run writes config.json, metrics.jsonl, ckpt.pt and sample.txt to <out_dir>/<name>.
"""
import argparse
import json
import time
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import save_checkpoint
from ..config import ModelConfig
from ..data.batches import VALIDATION_SEED, get_batch, load_meta, load_split, sample_offsets
from ..data.datasets import prepare_dataset
from ..data.tokenizers import get_tokenizer
from ..model.transformer import MoETransformer
from .evaluate import eval_record
from .metrics import format_eval, format_train, routing_stats, write_jsonl
from .optimizer import build_optimizer, lr_at
from .runtime import allow_tf32, autocast_context, pick_device, pick_dtype, pick_impl


def build_model(args: argparse.Namespace, vocab_size: int, device: str) -> tuple[MoETransformer, ModelConfig]:
    """Builds the model and its ModelConfig from the train.py flags, on device."""
    cfg = ModelConfig(
        vocab_size=vocab_size, seq_len=args.seq_len, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, ffn_type=args.ffn,
        n_routed_experts=args.n_experts, expert_hidden=args.expert_hidden, top_k=args.top_k,
        n_shared_experts=args.n_shared, first_k_dense=args.first_k_dense, router=args.router,
        aux_loss_coef=args.aux_loss_coef, bias_update_rate=args.bias_update_rate,
        capacity_factor=args.capacity_factor, moe_impl=args.impl, dense_hidden=args.dense_hidden or None,
    )
    return MoETransformer(cfg).to(device), cfg


def train_step(
    model: MoETransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    train_data: np.ndarray,
    rng: np.random.Generator,
    device: str,
    ctx: AbstractContextManager,
    lr: float,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """One optimizer step with gradient accumulation.

    Returns (loss, info, grad_norm), with loss, info["ce"] and info["aux"] averaged over the micro-batches
    and info["routing"] from the last micro-batch.
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_sum = ce_sum = aux_sum = 0.0
    for _ in range(args.grad_accum):
        offsets = sample_offsets(rng, train_data, args.batch_size, args.seq_len)
        x, y = get_batch(train_data, offsets, args.seq_len, device)
        with ctx:
            _, loss, info = model(x, y)
        scaler.scale(loss / args.grad_accum).backward()
        loss_sum += loss.detach()
        ce_sum += info["ce"]
        aux_sum += info["aux"]
    loss = loss_sum / args.grad_accum
    info = {"ce": ce_sum / args.grad_accum, "aux": aux_sum / args.grad_accum, "routing": info["routing"]}
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    model.update_router_biases()      # changes only aux-free routers
    return loss, info, grad_norm


def write_sample(
    model: MoETransformer,
    args: argparse.Namespace,
    run_dir: Path,
    device: str,
    ctx: AbstractContextManager,
    max_new_tokens: int = 400,
) -> str:
    """Generates a short sample with causal routing, saves it as sample.txt and returns it."""
    tokenizer = get_tokenizer(args.tokenizer)
    model.eval()
    prompt = "ROMEO:" if args.dataset == "shakespeare" and not args.data_dir else "Once upon a time"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    with ctx:
        out = model.generate(prompt_ids, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0].tolist())
    (run_dir / "sample.txt").write_text(text)
    return text


def train(args: argparse.Namespace) -> None:
    """Runs one training job from the train.py flags.

    --seed seeds torch, which draws the initial weights and the sample text, and numpy, which picks the batches.
    """
    device = pick_device(args.device)
    args.impl = pick_impl(args.impl, device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dtype = pick_dtype(args.dtype, device)
    ctx = autocast_context(device, dtype)
    allow_tf32(device)

    data_dir = args.data_dir or prepare_dataset(args.dataset, args.tokenizer, args.max_mb)
    meta = load_meta(data_dir)
    args.tokenizer = meta["tokenizer"]   # the data decides the tokenizer, not the flag
    train_data, val_data = load_split(data_dir, "train"), load_split(data_dir, "val")
    val_rng = np.random.default_rng(VALIDATION_SEED)
    val_offsets = [sample_offsets(val_rng, val_data, args.batch_size, args.seq_len) for _ in range(args.eval_iters)]

    model, cfg = build_model(args, meta["vocab_size"], device)
    total, active = model.num_params()
    name = args.name or (f"moe_{args.router}" if args.ffn == "moe" else "dense")
    run_dir = Path(args.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"model": asdict(cfg), "args": vars(args)}, indent=2))
    print(f"[run] {name} | device={device} dtype={dtype} impl={args.impl if args.ffn == 'moe' else '-'}")
    print(f"[run] params: total={total / 1e6:.2f}M  active/token={active / 1e6:.2f}M")

    optimizer = build_optimizer(model, args.weight_decay, args.lr, device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)   # fp16 needs loss scaling, bf16 doesn't
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len

    with open(run_dir / "metrics.jsonl", "w") as log_file:
        timer_start, timed_steps = time.time(), 0
        for step in range(args.steps + 1):
            if step % args.eval_interval == 0 or step == args.steps:
                record = eval_record(model, args, val_data, val_offsets, device, ctx, step)
                print(format_eval(record))
                write_jsonl(log_file, record)
                save_checkpoint(run_dir / "ckpt.pt", model, cfg, vars(args), step)
                timer_start, timed_steps = time.time(), 0   # don't count eval time in throughput
            if step == args.steps:
                break

            lr = lr_at(step, args)
            loss, info, grad_norm = train_step(model, optimizer, scaler, args, train_data, rng, device, ctx, lr)
            timed_steps += 1

            if step % args.log_interval == 0:
                if device == "cuda":
                    torch.cuda.synchronize()   # CUDA runs asynchronously, so wait for queued work before timing
                record = {"step": step, "train_loss": loss.item(), "ce": info["ce"].item(),
                          "aux": info["aux"].item(), "lr": lr, "grad_norm": float(grad_norm),
                          "tok_per_s": tokens_per_step * timed_steps / (time.time() - timer_start)}
                record.update(routing_stats(info["routing"]))
                print(format_train(record))
                write_jsonl(log_file, record)
                timer_start, timed_steps = time.time(), 0

    print("\n----- sample -----\n" + write_sample(model, args, run_dir, device, ctx))
