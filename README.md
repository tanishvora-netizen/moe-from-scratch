# moe-from-scratch

[![tests](https://github.com/tanishvora-netizen/moe-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/tanishvora-netizen/moe-from-scratch/actions/workflows/tests.yml)

A small Mixture-of-Experts language model I wrote in PyTorch to understand how DeepSeek-style MoE layers work,
including the parts that usually hide inside a library: a Triton kernel for the expert matmuls, and the
load-balancing tricks that stop the router from sending everything to a few experts.

What's in it:

- A ~7M parameter decoder with DeepSeek-MoE style layers: 16 small routed experts (each token picks 4) plus 2
  shared experts that every token goes through.
- A Triton grouped-GEMM kernel, forward and backward, that runs every expert's matmul in one launch instead of a
  Python loop over experts.
- Three load-balancing strategies behind one flag (`--router`), and a dense baseline with the same compute per
  token to compare them against.
- A script that measures which tokens each expert ends up handling.

## Status

The model, the three routers and the kernel are done and tested on CPU. CI runs the Triton kernels in Triton's
interpreter and checks them against plain PyTorch, which tests correctness but not speed. The GPU part (the
ablation runs and the kernel timings) hasn't been run yet, so there are no results here yet. That's next.

## Setup

```bash
git clone https://github.com/tanishvora-netizen/moe-from-scratch.git
cd moe-from-scratch
pip install -e ".[all]"
pytest -q
```

Without an NVIDIA GPU, the tests run the kernel in Triton's CPU interpreter and training uses the plain PyTorch
path. A short run on Shakespeare that finishes on a laptop:

```bash
python scripts/train.py --dataset shakespeare --steps 200 --batch_size 16 --eval_interval 100 --impl naive
```

## Running the experiment (one GPU)

```bash
python scripts/benchmark.py     # checks Triton against the naive path, then times naive vs Triton vs dense
bash scripts/run_ablations.sh   # dense + the 3 routers, then the comparison plot and expert analysis
```

Or one step at a time:

```bash
python scripts/train.py --name dense             --ffn dense
python scripts/train.py --name moe_aux_loss      --router aux_loss
python scripts/train.py --name moe_aux_free      --router aux_free
python scripts/train.py --name moe_expert_choice --router expert_choice
python scripts/plot_runs.py runs/dense runs/moe_aux_loss runs/moe_aux_free runs/moe_expert_choice
python scripts/analyze_experts.py --ckpt runs/moe_aux_free/ckpt.pt
python scripts/sample.py --ckpt runs/moe_aux_free/ckpt.pt --prompt "Once upon a time"
```

Each run writes `runs/<name>/` with `config.json`, `metrics.jsonl`, `ckpt.pt` and a text sample, and
`plot_runs.py` prints the final numbers as a Markdown table.

Defaults: the first 200 MB of TinyStories, byte-level tokens, context 256, batch 64, 5000 steps (about 82M
tokens), AdamW with warmup and cosine decay, bf16 autocast. `--data_dir` trains on any directory that has
`meta.json`, `train.bin` and `val.bin`.

## Model

| | MoE (default) | Dense baseline |
|---|---|---|
| d_model / layers / heads | 256 / 6 / 8 | 256 / 6 / 8 |
| FFN | 16 routed experts (hidden 64), top-4, + 2 shared | SwiGLU, hidden 384 |
| Total params | 6.97M | 3.41M |
| Active params per token | 3.44M | 3.41M |

The dense hidden size is (2 shared + 4 routed) x 64 = 384, so both models do about the same FFN compute per
token. Any gap in loss comes from the MoE having more experts to choose from, not from doing more work.

## Load balancing

Left alone, a router tends to send most tokens to a few experts, and the others stop learning. The options:

| `--router` | How it balances | From |
|---|---|---|
| `aux_loss` | softmax top-k plus a loss `E * sum_e(f_e * P_e)` with weight 0.01. It's 1 when load is even and E/k when every token goes to the same k experts | Switch Transformer, GShard |
| `aux_free` | sigmoid scores. A per-expert bias is added only when picking the top k, and after each step it moves by +/- gamma toward experts that got fewer tokens. No extra loss | DeepSeek-V3 |
| `expert_choice` | each expert picks its top C tokens, so load is even by construction | Zhou et al. 2022 |

Expert choice has a catch in a language model. Whether a token gets picked depends on the scores of the other
tokens in the batch, including later tokens in the same sequence, so during training the model gets to look at
the future and its validation loss comes out better than it should. For these runs `train.py` also logs
`val_loss_token_choice`, which evaluates the same model with ordinary per-token top-k routing. The gap between
the two numbers is how much it was cheating. `tests/test_router.py` checks the leak directly: changing later
tokens changes earlier logits under expert choice, and doesn't under the other two routers.

What gets logged to `metrics.jsonl`:

- `val_loss` on fixed validation windows, the same windows for every run
- `maxvio` = (max expert load - mean load) / mean load, averaged over layers, where 0 is perfectly even. This is
  the metric from the auxiliary-loss-free paper
- per-layer `loads`, `dropped_frac` for expert choice (tokens that no expert picked), and `tok_per_s`

`analyze_experts.py` adds heatmaps and a normalized mutual information, I(token; expert) / H(expert), per layer,
as one number for how specialized the experts are. Byte-level tokens keep the heatmaps easy to read, since the
categories are things like spaces, vowels, capitals and punctuation.

## The Triton kernel

After routing, each token has k (token, expert) pairs, and the expert part of the layer is:

```
sort pairs by expert -> grouped GEMM 1 (x @ W_gate_up[e], gather fused in)
                     -> SwiGLU -> grouped GEMM 2 (@ W_down[e]) -> scale by gate, add back to each token
```

The naive path (`--impl naive`) loops over the 16 experts in Python and does a gather, two small matmuls and a
scatter for each one. That's on the order of a hundred small kernel launches per layer, plus a host sync to read
the expert offsets.

The Triton path (`moe/kernels.py`) does each GEMM in one launch:

- Each program computes a 64-row output tile, and tiles never cross an expert boundary, so a tile only reads one
  expert's weights.
- A tile schedule maps each tile to its expert and row range. It's built on the GPU with `searchsorted`, and the
  kernel is launched with an upper bound of `ceil(M / 64) + E` tiles, so the CPU never waits to find out how many
  tiles there really are. The extra tiles return immediately.
- Tiles read token rows straight from the unpermuted activations, so the sorted copy of `x` is never written out.
- Backward is Triton too. `dX` reuses the forward kernel with the weights read transposed through their strides,
  and `dW[e] = X_e^T dY_e` is a second kernel that sums over each expert's rows.
- `BLOCK_N`, `BLOCK_K`, warps and stages are autotuned per shape.

One thing that has to be handled by hand: `torch.autocast` doesn't cast the inputs of a custom
`autograd.Function`, so `GroupedGEMM` looks up the active autocast dtype itself and runs in bf16 when the rest of
the model does.

## Layout

```
moe/
  config.py      ModelConfig, every hyperparameter, validated when it's created
  layers.py      RMSNorm, causal attention with RoPE, SwiGLU
  router.py      the three routers, all producing the same (token, expert) layout
  kernels.py     Triton grouped-GEMM kernels and the autograd wrapper
  moe_layer.py   the MoE layer, with the Triton path and the naive reference path
  model.py       the transformer, parameter counting, checkpoint loading
  data.py        Shakespeare / TinyStories download and tokenization
scripts/
  train.py, benchmark.py, analyze_experts.py, plot_runs.py, sample.py, run_ablations.sh, smoke_test.sh
tests/
  test_kernel.py                  kernel vs a PyTorch reference, forward and both gradients
  test_moe.py                     Triton layer matches the naive layer for every router, model overfits a batch
  test_router.py                  routing layout, balance-loss values, causality
  test_checkpoint_and_config.py   checkpoint round trip, config validation
  test_train_script.py            train.py end to end, then sample / analyze / plot on its output
```

## Next

- [ ] Run the ablation on a GPU and put the results and plots here
- [ ] Compare what the experts specialize in under each router
- [ ] Kernel: fuse SwiGLU into the first GEMM, and the gate scaling and scatter-add into the second
- [ ] Split the weight-gradient kernel along the token dimension, so one overloaded expert doesn't turn into the
      slowest program in the launch
- [ ] Try the aux-loss-free bias outside MoE, e.g. to even out slot usage in learned hash-table encoders, where
      many inputs competing for one slot is the same problem as tokens piling onto one expert

## References

- Fedus, Zoph, Shazeer. [Switch Transformers](https://arxiv.org/abs/2101.03961). 2021.
- Lepikhin et al. [GShard](https://arxiv.org/abs/2006.16668). 2020.
- Dai et al. [DeepSeekMoE](https://arxiv.org/abs/2401.06066). 2024.
- Wang et al. [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664). 2024.
- DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). 2024.
- Zhou et al. [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368). 2022.
- Eldan, Li. [TinyStories](https://arxiv.org/abs/2305.07759). 2023.

## License

MIT
