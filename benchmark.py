"""
Benchmarks (needs an NVIDIA GPU):

  1. correctness  - Triton MoE vs naive MoE: outputs and gradients must match
  2. kernel only  - one grouped-GEMM launch vs a Python loop of per-expert matmuls
  3. full layer   - naive MoE vs Triton MoE vs matched dense SwiGLU, fwd and fwd+bwd

  python benchmark.py
  python benchmark.py --d_model 1024 --expert_hidden 512 --n_experts 64 --top_k 6   # bigger shapes
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch
import triton

from moe.config import ModelConfig
from moe.kernels import grouped_gemm
from moe.layers import SwiGLU
from moe.moe_layer import MoE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--expert_hidden", type=int, default=64)
    p.add_argument("--n_experts", type=int, default=16)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--n_shared", type=int, default=2)
    p.add_argument("--tokens", default="2048,8192,16384,32768,65536")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--out", default="benchmarks")
    return p.parse_args()


def call(m, x):
    out = m(x)
    return out[0] if isinstance(out, tuple) else out


def bench(fn, grad_to_none=None):
    return triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=grad_to_none)


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "benchmark.py needs a CUDA GPU"
    dev, dtype = "cuda", {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(0)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    cfg = ModelConfig(d_model=args.d_model, n_heads=1, expert_hidden=args.expert_hidden,
                      n_routed_experts=args.n_experts, top_k=args.top_k,
                      n_shared_experts=args.n_shared, router="aux_loss", moe_impl="triton")
    moe_t = MoE(cfg).to(dev)
    moe_t.reset_parameters(0.02, 0.02)
    moe_n = MoE(replace(cfg, moe_impl="naive")).to(dev)
    moe_n.load_state_dict(moe_t.state_dict())
    dense = SwiGLU(cfg.d_model, cfg.matched_dense_hidden).to(dev)
    print(f"d={cfg.d_model} E={cfg.n_routed_experts} k={cfg.top_k} h={cfg.expert_hidden} "
          f"shared={cfg.n_shared_experts} | dense baseline hidden={cfg.matched_dense_hidden} | {args.dtype}")

    # ---------------------------------------------------------------- 1. correctness
    x0 = torch.randn(4096, cfg.d_model, device=dev)
    outs, xgrads = [], []
    for m in (moe_t, moe_n):
        x = x0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype):
            y = call(m, x)
        y.float().pow(2).sum().backward()
        outs.append(y.detach())
        xgrads.append(x.grad)
    y_t, y_n = outs
    gx_t, gx_n = xgrads

    def rel(a, b):
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()

    print("\n[1] correctness (relative error, Triton vs naive; bf16 noise ~1e-2 is fine)")
    print(f"    output     {rel(y_t, y_n):.2e}")
    print(f"    dL/dx      {rel(gx_t, gx_n):.2e}")
    print(f"    dL/dW_up   {rel(moe_t.w_gate_up.grad, moe_n.w_gate_up.grad):.2e}")
    print(f"    dL/dW_down {rel(moe_t.w_down.grad, moe_n.w_down.grad):.2e}")
    print(f"    dL/dRouter {rel(moe_t.router.gate.weight.grad, moe_n.router.gate.weight.grad):.2e}")

    tokens = [int(t) for t in args.tokens.split(",")]
    rows = []

    # ---------------------------------------------------------------- 2. kernel only
    print("\n[2] kernel only: up-projection GEMM over all experts (ms)")
    print(f"    {'tokens':>8} | {'torch loop':>10} | {'triton':>8} | speedup")
    E, d, h2 = cfg.n_routed_experts, cfg.d_model, 2 * cfg.expert_hidden
    W = torch.randn(E, d, h2, device=dev, dtype=dtype) * 0.02
    for T in tokens:
        M = T * cfg.top_k
        expert = torch.randint(0, E, (M,), device=dev).sort().values
        offsets = torch.nn.functional.pad(torch.bincount(expert, minlength=E).cumsum(0), (1, 0))
        A = torch.randn(M, d, device=dev, dtype=dtype)
        offs = offsets.tolist()
        loop = lambda: torch.cat([A[offs[e]:offs[e + 1]] @ W[e] for e in range(E)])
        tri = lambda: grouped_gemm(A, W, offsets)
        torch.testing.assert_close(tri(), loop(), atol=2e-2, rtol=2e-2)
        t_loop, t_tri = bench(loop), bench(tri)
        print(f"    {T:>8} | {t_loop:>10.3f} | {t_tri:>8.3f} | {t_loop / t_tri:5.2f}x")
        rows.append({"section": "kernel", "tokens": T, "impl": "torch_loop", "ms": t_loop})
        rows.append({"section": "kernel", "tokens": T, "impl": "triton", "ms": t_tri})

    # ---------------------------------------------------------------- 3. full layer
    print("\n[3] full FFN layer incl. routing (ms)")
    print(f"    {'tokens':>8} | {'naive fwd':>9} | {'triton fwd':>10} | {'dense fwd':>9} || "
          f"{'naive f+b':>9} | {'triton f+b':>10} | {'dense f+b':>9}")
    for T in tokens:
        x = torch.randn(T, cfg.d_model, device=dev, requires_grad=True)
        res = {}
        for name, m in [("naive", moe_n), ("triton", moe_t), ("dense", dense)]:
            def fwd(m=m):
                with torch.autocast("cuda", dtype=dtype):
                    return call(m, x)

            def fwd_bwd(m=m):
                with torch.autocast("cuda", dtype=dtype):
                    y = call(m, x)
                y.float().sum().backward()

            res[name, "fwd"] = bench(fwd)
            res[name, "fwd_bwd"] = bench(fwd_bwd, grad_to_none=[x, *m.parameters()])
            for kind in ("fwd", "fwd_bwd"):
                rows.append({"section": "layer_" + kind, "tokens": T, "impl": name, "ms": res[name, kind]})
        print(f"    {T:>8} | {res['naive', 'fwd']:>9.3f} | {res['triton', 'fwd']:>10.3f} | "
              f"{res['dense', 'fwd']:>9.3f} || {res['naive', 'fwd_bwd']:>9.3f} | "
              f"{res['triton', 'fwd_bwd']:>10.3f} | {res['dense', 'fwd_bwd']:>9.3f}")

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "tokens", "impl", "ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out_dir / 'results.csv'}")

    try:
        import matplotlib.pyplot as plt
        sections = ["kernel", "layer_fwd", "layer_fwd_bwd"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, sec in zip(axes, sections):
            for impl in sorted({r["impl"] for r in rows if r["section"] == sec}):
                pts = [(r["tokens"], r["ms"]) for r in rows if r["section"] == sec and r["impl"] == impl]
                ax.plot(*zip(*pts), marker="o", label=impl)
            ax.set(title=sec, xlabel="tokens", ylabel="ms", xscale="log", yscale="log")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "results.png", dpi=150)
        print(f"saved {out_dir / 'results.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
