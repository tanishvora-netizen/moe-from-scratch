# MoE from scratch: DeepSeek-style experts + a custom Triton kernel

A ~7M-parameter Mixture-of-Experts transformer written from scratch in PyTorch, with:

- **DeepSeek-MoE architecture**: 16 fine-grained routed experts (top-4) plus 2 always-on shared experts per layer
- **A custom Triton grouped-GEMM kernel** (forward *and* backward) that runs all experts in one launch, benchmarked against a naive Python loop and a matched dense baseline
- **A load-balancing ablation**: Switch-style auxiliary loss vs DeepSeek-V3 auxiliary-loss-free bias vs expert-choice routing
- **Expert-specialization analysis**: which tokens each expert ends up handling

| | MoE (default) | Dense baseline |
|---|---|---|
| d_model / layers / heads | 256 / 6 / 8 | 256 / 6 / 8 |
| FFN | 16 routed experts (h=64), top-4, + 2 shared experts | SwiGLU, h=384 |
| Total params | ~7.0M | ~3.4M |
| Active params / token | ~3.4M | ~3.4M |

The dense FFN width is chosen so both models do the same compute per token. That way the comparison isolates what the extra (inactive) expert capacity buys you.

## Project layout

```
moe/
  config.py       all hyperparameters (ModelConfig)
  layers.py       RMSNorm, RoPE causal attention, SwiGLU
  router.py       aux_loss / aux_free / expert_choice routing -> sorted (token, expert) layout
  kernels.py      Triton grouped-GEMM kernels + autograd Function   <- the custom kernel
  moe_layer.py    MoE layer: naive loop path and Triton path, same weights
  model.py        full transformer, param counting, checkpoint loading
data.py           Shakespeare / TinyStories download + byte or GPT-2 tokenization
train.py          training loop, logs loss + load-balance metrics to runs/<name>/metrics.jsonl
benchmark.py      correctness check + kernel/layer timings (naive vs Triton vs dense)
analyze_experts.py  expert specialization heatmaps + mutual-information score
plot_runs.py      ablation comparison plot
sample.py         generate text from a checkpoint
tests/            kernel and layer correctness tests (run on CPU via the Triton interpreter)
scripts/          smoke_test.sh, run_ablations.sh
```

## 1. Open it in VS Code

1. Unzip, then **File → Open Folder…** and pick `moe-from-scratch`.
2. Open a terminal (**Ctrl+`**) and create an environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Ctrl+Shift+P → "Python: Select Interpreter"** → choose `.venv`.
4. The Run and Debug panel already has launch configs (smoke test, train, benchmark, analyze), and the Testing panel picks up `tests/`.

> Triton only runs on Linux with an NVIDIA GPU. On a Mac or Windows laptop, everything except the Triton path still works: use `--impl naive` locally, and do the real runs on a rented GPU (section 3).

## 2. Smoke test (any machine, ~1 min)

```bash
pytest -q                     # kernel tests run in Triton's CPU interpreter if there's no GPU (needs triton installed)
python train.py --name smoke --dataset shakespeare --steps 200 --batch_size 16 --eval_interval 100 --impl naive
```

## 3. Real runs on a rented GPU

Any single A10 / L4 / 4090 / A100 / H100 is plenty (RunPod, Lambda, Vast all work).

1. In VS Code, install the **Remote - SSH** extension, then **Ctrl+Shift+P → "Remote-SSH: Connect to Host…"** and paste the SSH command your provider gives you.
2. Get the code onto the box: push this folder to GitHub and `git clone` it there, or drag the folder into the remote VS Code explorer.
3. On the box:
   ```bash
   pip install -r requirements.txt
   pytest -q                         # kernel vs PyTorch reference on the real GPU
   python benchmark.py               # Triton vs naive vs dense
   bash scripts/run_ablations.sh     # all 4 training runs + plots + analysis
   ```

Individual commands:

```bash
python train.py --name dense             --ffn dense
python train.py --name moe_aux_loss      --router aux_loss
python train.py --name moe_aux_free      --router aux_free
python train.py --name moe_expert_choice --router expert_choice
python plot_runs.py runs/dense runs/moe_aux_loss runs/moe_aux_free runs/moe_expert_choice
python analyze_experts.py --ckpt runs/moe_aux_free/ckpt.pt
python sample.py --ckpt runs/moe_aux_free/ckpt.pt --prompt "Once upon a time"
```

Defaults: TinyStories (first 200 MB), byte-level tokens, seq_len 256, batch 64, 5000 steps (~82M tokens), AdamW with cosine LR, bf16 autocast.

## How the Triton kernel works

After routing, each token has k (token, expert) assignments. The MoE forward pass is:

```
router  ->  sort assignments by expert  ->  grouped GEMM #1 (x @ W_gate_up[e], gather fused)
        ->  SwiGLU  ->  grouped GEMM #2 (@ W_down[e])  ->  multiply by gate, scatter-add to tokens
```

**Naive version** (`moe_impl="naive"`): a Python loop over experts, each doing a gather, two small matmuls and a scatter. With 16 experts that adds up to over a hundred small kernel launches per layer, plus a host sync. The individual matmuls are tiny, so the GPU sits mostly idle.

**Triton version** (`moe/kernels.py`): one kernel launch covers every expert.
- Each program computes a 64-row × BLOCK_N output tile. A tile never crosses an expert boundary, so it only loads one expert's weights.
- A **tile schedule** maps each tile to (expert, row range). It's built with `searchsorted` on the GPU and launched with an upper bound of tiles, so there is **no host sync**. Surplus tiles exit immediately.
- The **gather is fused**: tiles load token rows straight from the un-permuted activations via `row_idx`, so the permuted copy is never written to memory.
- **Backward is also Triton**: `dX` reuses the same kernel with the weights read transposed through strides; `dW[e] = Xₑᵀ dYₑ` uses a second kernel that reduces over each expert's rows.
- BLOCK_N / BLOCK_K / warps / stages are autotuned per shape.

Ideas if you want to push the kernel further: fuse SwiGLU into GEMM #1's epilogue, fuse the gate-multiply + scatter-add into GEMM #2's epilogue (atomic adds), or make it a persistent kernel.

## The load-balancing ablation

| `--router` | How it balances | Paper |
|---|---|---|
| `aux_loss` | softmax top-k + loss `E · Σ fₑ·Pₑ` (coef 0.01) | Switch Transformer / GShard |
| `aux_free` | sigmoid scores; a per-expert bias is added **only for picking** experts and nudged by `±γ` each step toward under-loaded experts; no extra loss term | DeepSeek-V3 (Wang et al. 2024) |
| `expert_choice` | each expert picks its top-C tokens, so load is perfectly balanced by construction | Zhou et al. 2022 |

**Important caveat for expert choice in a language model:** whether a token gets picked depends on the scores of the *other* tokens in the batch, including tokens later in the same sequence. Training is therefore not strictly causal, and its validation loss can look better than it really is. To make this visible, `train.py` also logs `val_loss_token_choice` for expert-choice runs, which re-evaluates the same model with causal per-token top-k routing. The gap between the two numbers is a nice finding to put in your writeup.

### Metrics logged (runs/&lt;name&gt;/metrics.jsonl)

- `val_loss` (and `val_loss_token_choice` for expert choice), measured on fixed validation windows so runs are directly comparable
- `maxvio` = (max expert load − mean load) / mean load, averaged over layers (0 = perfectly balanced). This is the metric used in the aux-loss-free paper.
- `loads`: per-layer expert load fractions
- `dropped_frac` (expert choice only): fraction of tokens that got zero routed experts
- `tok_per_s`: training throughput

`analyze_experts.py` adds **normalized mutual information** I(token; expert) / H(expert) per layer, a single "how specialized are the experts" number to compare across routers. With byte-level tokens the heatmaps are easy to read: watch for experts that specialize in spaces, vowels, capitals, or punctuation.
