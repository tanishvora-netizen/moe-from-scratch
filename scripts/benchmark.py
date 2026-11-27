"""
Benchmarks (needs an NVIDIA GPU):

  1. correctness  - Triton MoE vs naive MoE: outputs and gradients must match
  2. kernel only  - one grouped-GEMM launch vs a Python loop of per-expert matmuls
  3. full layer   - naive MoE vs Triton MoE vs matched dense SwiGLU, fwd and fwd+bwd

  python scripts/benchmark.py
  python scripts/benchmark.py --d_model 1024 --expert_hidden 512 --n_experts 64 --top_k 6   # bigger shapes

Shapes: T tokens, E routed experts, M = T * top_k assignments, d the model width, h the expert hidden size.
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch
import triton

from moe.config import ModelConfig
from moe.experts.layer import MoELayer
from moe.kernels.grouped_gemm import grouped_gemm
from moe.model.swiglu import SwiGLU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--expert_hidden", type=int, default=64)
    parser.add_argument("--n_experts", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--n_shared", type=int, default=2)
    parser.add_argument("--tokens", default="2048,8192,16384,32768,65536")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--out", default="benchmarks")
    return parser.parse_args()


def layer_output(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """MoELayer returns (y, routing) and SwiGLU returns y; this returns y for both."""
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


def bench(fn, grad_to_none=None) -> float:
    """Milliseconds per call of fn."""
    return triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=grad_to_none)


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


def check_correctness(moe_triton: MoELayer, moe_naive: MoELayer, d_model: int, dtype: torch.dtype, device: str) -> None:
    """Section 1: one forward and backward through both MoE layers, then the relative error of every result."""
    x0 = torch.randn(4096, d_model, device=device)
    outs, x_grads = [], []
    for layer in (moe_triton, moe_naive):
        x = x0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype):
            y = layer_output(layer, x)
        y.float().pow(2).sum().backward()
        outs.append(y.detach())
        x_grads.append(x.grad)

    print("\n[1] correctness (relative error, Triton vs naive; bf16 noise ~1e-2 is fine)")
    print(f"    output     {relative_error(outs[0], outs[1]):.2e}")
    print(f"    dL/dx      {relative_error(x_grads[0], x_grads[1]):.2e}")
    print(f"    dL/dW_up   {relative_error(moe_triton.w_gate_up.grad, moe_naive.w_gate_up.grad):.2e}")
    print(f"    dL/dW_down {relative_error(moe_triton.w_down.grad, moe_naive.w_down.grad):.2e}")
    print(f"    dL/dRouter {relative_error(moe_triton.router.gate.weight.grad, moe_naive.router.gate.weight.grad):.2e}")


def time_kernel(cfg: ModelConfig, tokens: list[int], dtype: torch.dtype, device: str) -> list[dict]:
    """Section 2: the up-projection GEMM over all experts, one grouped_gemm launch vs a loop of per-expert matmuls."""
    print("\n[2] kernel only: up-projection GEMM over all experts (ms)")
    print(f"    {'tokens':>8} | {'torch loop':>10} | {'triton':>8} | speedup")
    E, d, h = cfg.n_routed_experts, cfg.d_model, cfg.expert_hidden
    weights = torch.randn(E, d, 2 * h, device=device, dtype=dtype) * 0.02
    rows = []
    for T in tokens:
        M = T * cfg.top_k
        expert = torch.randint(0, E, (M,), device=device).sort().values
        offsets = torch.nn.functional.pad(torch.bincount(expert, minlength=E).cumsum(0), (1, 0))
        activations = torch.randn(M, d, device=device, dtype=dtype)   # one row per assignment, sorted by expert
        bounds = offsets.tolist()
        torch_loop = lambda x=activations, bounds=bounds: torch.cat(
            [x[bounds[e]:bounds[e + 1]] @ weights[e] for e in range(E)])
        triton_gemm = lambda x=activations, offsets=offsets: grouped_gemm(x, weights, offsets)
        torch.testing.assert_close(triton_gemm(), torch_loop(), atol=2e-2, rtol=2e-2)
        loop_ms, triton_ms = bench(torch_loop), bench(triton_gemm)
        print(f"    {T:>8} | {loop_ms:>10.3f} | {triton_ms:>8.3f} | {loop_ms / triton_ms:5.2f}x")
        rows.append({"section": "kernel", "tokens": T, "impl": "torch_loop", "ms": loop_ms})
        rows.append({"section": "kernel", "tokens": T, "impl": "triton", "ms": triton_ms})
    return rows


def time_layers(
    layers: dict[str, torch.nn.Module], d_model: int, tokens: list[int], dtype: torch.dtype, device: str
) -> list[dict]:
    """Section 3: each whole FFN layer, routing included, forward and forward+backward. layers maps name to layer."""
    print("\n[3] full FFN layer incl. routing (ms)")
    print(f"    {'tokens':>8} | {'naive fwd':>9} | {'triton fwd':>10} | {'dense fwd':>9} || "
          f"{'naive f+b':>9} | {'triton f+b':>10} | {'dense f+b':>9}")
    rows = []
    for T in tokens:
        x = torch.randn(T, d_model, device=device, requires_grad=True)
        ms = {}
        for name, layer in layers.items():
            def fwd(layer=layer, x=x):
                with torch.autocast("cuda", dtype=dtype):
                    return layer_output(layer, x)

            def fwd_bwd(layer=layer, x=x):
                with torch.autocast("cuda", dtype=dtype):
                    y = layer_output(layer, x)
                y.float().sum().backward()

            ms[name, "fwd"] = bench(fwd)
            ms[name, "fwd_bwd"] = bench(fwd_bwd, grad_to_none=[x, *layer.parameters()])
            for kind in ("fwd", "fwd_bwd"):
                rows.append({"section": "layer_" + kind, "tokens": T, "impl": name, "ms": ms[name, kind]})
        print(f"    {T:>8} | {ms['naive', 'fwd']:>9.3f} | {ms['triton', 'fwd']:>10.3f} | "
              f"{ms['dense', 'fwd']:>9.3f} || {ms['naive', 'fwd_bwd']:>9.3f} | "
              f"{ms['triton', 'fwd_bwd']:>10.3f} | {ms['dense', 'fwd_bwd']:>9.3f}")
    return rows


def save_results(rows: list[dict], out_dir: Path) -> None:
    """results.csv, plus results.png when matplotlib is installed."""
    with open(out_dir / "results.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["section", "tokens", "impl", "ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {out_dir / 'results.csv'}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sections = ["kernel", "layer_fwd", "layer_fwd_bwd"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, section in zip(axes, sections, strict=True):
        for impl in sorted({row["impl"] for row in rows if row["section"] == section}):
            points = [(row["tokens"], row["ms"]) for row in rows if row["section"] == section and row["impl"] == impl]
            ax.plot(*zip(*points, strict=True), marker="o", label=impl)
        ax.set(title=section, xlabel="tokens", ylabel="ms", xscale="log", yscale="log")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "results.png", dpi=150)
    print(f"saved {out_dir / 'results.png'}")


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "benchmark.py needs a CUDA GPU"
    device, dtype = "cuda", {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(0)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    cfg = ModelConfig(d_model=args.d_model, n_heads=1, expert_hidden=args.expert_hidden,
                      n_routed_experts=args.n_experts, top_k=args.top_k,
                      n_shared_experts=args.n_shared, router="aux_loss", moe_impl="triton")
    moe_triton = MoELayer(cfg).to(device)
    moe_triton.reset_parameters(0.02, 0.02)
    moe_naive = MoELayer(replace(cfg, moe_impl="naive")).to(device)
    moe_naive.load_state_dict(moe_triton.state_dict())
    dense = SwiGLU(cfg.d_model, cfg.matched_dense_hidden).to(device)
    print(f"d={cfg.d_model} E={cfg.n_routed_experts} k={cfg.top_k} h={cfg.expert_hidden} "
          f"shared={cfg.n_shared_experts} | dense baseline hidden={cfg.matched_dense_hidden} | {args.dtype}")

    # The sections run in this order so the seeded random inputs match across runs.
    check_correctness(moe_triton, moe_naive, cfg.d_model, dtype, device)
    tokens = [int(t) for t in args.tokens.split(",")]
    rows = time_kernel(cfg, tokens, dtype, device)
    rows += time_layers({"naive": moe_naive, "triton": moe_triton, "dense": dense}, cfg.d_model, tokens, dtype, device)
    save_results(rows, out_dir)


if __name__ == "__main__":
    main()
