# moe-from-scratch

[![tests](https://github.com/tanishvora-netizen/moe-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/tanishvora-netizen/moe-from-scratch/actions/workflows/tests.yml)

A small Mixture-of-Experts language model built from scratch in PyTorch, with a custom Triton GPU kernel for the experts. It trains a 7M-parameter model on short children's stories and compares three ways to keep the experts evenly used.

## What it is

A Mixture-of-Experts (MoE) layer replaces a transformer's feed-forward network with many small networks called experts. A router sends each token to a few of them. The model holds more parameters than any one token uses, so it gets more capacity for the same compute per token.

This repo builds that from the ground up:

- A 6-layer decoder with DeepSeek-MoE style layers. Each layer has 16 small routed experts, and each token uses 4 of them plus 2 shared experts that every token goes through.
- Three load-balancing routers, chosen with `--router`.
- A Triton kernel that runs every expert's matrix multiply in one GPU launch, for the forward and the backward pass.
- A dense baseline that does the same compute per token, to compare against.
- Scripts that train the models, compare runs, sample text and measure what each expert ends up handling.

## Status

The model, the routers and the kernel are finished and tested on CPU. The GPU experiments haven't been run, so the results section is empty for now.

## How it works

### The MoE layer

| | MoE (default) | Dense baseline |
|---|---|---|
| Feed-forward | 16 routed experts of width 64, top 4, plus 2 shared | one SwiGLU of width 384 |
| Total parameters | 6.97M | 3.41M |
| Parameters used per token | 3.44M | 3.41M |

The dense width is (2 shared + 4 routed) x 64 = 384. Both models do about the same work per token, so a gap in loss comes from the MoE having more experts to pick from.

### Keeping the experts balanced

A router left alone tends to send most tokens to a few experts, and the rest stop learning. Each router handles this differently.

- `aux_loss` picks each token's top 4 experts from a softmax. It adds a loss term that grows as the load gets uneven. The term is 1.0 when every expert gets the same share and 4.0 when every token picks the same 4 experts. This is the method from Switch Transformer and GShard.
- `aux_free` scores experts with a sigmoid. Each expert has a bias that is added only when picking the top 4. After every optimizer step, the bias goes up for experts that got too few tokens and down for the others. There is no extra loss. This is the method from DeepSeek-V3.
- `expert_choice` lets each expert pick the tokens it scores highest, instead of each token picking experts. Every expert gets the same number of tokens by construction. This is the method from Zhou et al. (2022).

Expert choice has a catch in a language model. Whether a token gets picked depends on the other tokens in the batch, including later tokens in the same sequence. The model can use that to see the future, so its validation loss looks better than it is. For expert-choice runs, `train.py` also logs `val_loss_token_choice`, the loss of the same model when each token picks its own top 4. The gap between the two numbers includes how much the model relied on future tokens. It also includes the cost of routing the model in a way it never trained with, so it can overstate the first. `tests/test_router.py` checks this directly. Under expert choice, changing later tokens changes the predictions for earlier ones. Under the other two routers, it doesn't.

### The Triton kernel

After routing, each token has 4 (token, expert) pairs. The pairs are sorted by expert, so each expert's tokens sit in one block of rows. The simple version, `--impl naive`, loops over the 16 experts in Python and runs two small matrix multiplies for each one. That is about a hundred small GPU launches per layer.

The Triton version in `moe/kernels/` runs each matrix multiply for all experts in one launch:

- Each GPU program computes one 64-row tile of output, and a tile never crosses from one expert to the next.
- The map from tiles to experts is built on the GPU, so the CPU never waits to learn how many tokens each expert got.
- Tiles read token rows straight from the unsorted activations, so the sorted copy is never written to memory.
- The backward pass is Triton too, with a second kernel for the weight gradients.

## Quick start on CPU

```bash
git clone https://github.com/tanishvora-netizen/moe-from-scratch.git
cd moe-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q
```

Without an NVIDIA GPU, the tests run the Triton kernels in Triton's CPU interpreter, and training uses the naive path. This run on Shakespeare finishes on a laptop:

```bash
python scripts/train.py --dataset shakespeare --steps 200 --batch_size 16 --eval_interval 100 --impl naive
```

## Running on an NVIDIA GPU

Triton only runs on Linux. On Windows, use WSL2.

1. On Windows only, install Ubuntu under WSL2 from PowerShell, then restart:

   ```bash
   wsl --install -d Ubuntu
   ```

   Keep the normal NVIDIA driver for Windows. Don't install a GPU driver inside Ubuntu.

2. Check that Linux sees the GPU:

   ```bash
   nvidia-smi
   ```

3. Follow the CPU quick start above. On Linux, `pip` installs a PyTorch build that includes CUDA and Triton.

With a GPU, `pytest -q` compiles and runs the real kernels and checks them against plain PyTorch.

## Experiments

This command checks the kernel, times it, trains the dense baseline and the three routers, and then compares them:

```bash
bash scripts/run_ablations.sh
```

The same steps, one at a time:

```bash
python scripts/benchmark.py
python scripts/train.py --name dense             --ffn dense
python scripts/train.py --name moe_aux_loss      --router aux_loss
python scripts/train.py --name moe_aux_free      --router aux_free
python scripts/train.py --name moe_expert_choice --router expert_choice
python scripts/plot_runs.py runs/dense runs/moe_aux_loss runs/moe_aux_free runs/moe_expert_choice
python scripts/analyze_experts.py --ckpt runs/moe_aux_free/ckpt.pt
python scripts/sample.py --ckpt runs/moe_aux_free/ckpt.pt --prompt "Once upon a time"
```

By default, each run trains for 5000 steps at batch 64 with context 256, about 82M tokens, on the first 200 MB of TinyStories. Tokens are single bytes. Each run writes `runs/<name>/` with `config.json`, `metrics.jsonl`, `ckpt.pt` and a text sample.

The logs record the validation loss, the load on each expert, and `maxvio`, the gap between the busiest expert's load and the average load, divided by the average. A `maxvio` of 0 means perfect balance. `analyze_experts.py` draws heatmaps of which kinds of bytes each expert handles, such as spaces, vowels or punctuation, and reports one number per layer for how specialized the experts are.

## Results

Not run yet. `plot_runs.py` prints a Markdown table that goes here.

## Code layout

```
moe/
  config.py          ModelConfig: every model setting, checked when it's created
  checkpoint.py      saving and loading ckpt.pt
  model/             norm, rope, attention, swiglu, block, transformer
  routing/           assignments (the layout all routers share), base, aux_loss, aux_free, expert_choice
  experts/           layer (the MoE layer), naive (the Python loop), fused (the Triton path)
  kernels/           tile_schedule, triton_kernels, grouped_gemm
  data/              tokenizers, datasets, batches
  training/          options, runtime, optimizer, evaluate, metrics, trainer
  analysis/          specialization, runs, plots
scripts/             train.py, benchmark.py, analyze_experts.py, plot_runs.py, sample.py,
                     run_ablations.sh, smoke_test.sh
tests/               the kernel against plain PyTorch, Triton against the naive path, the routers,
                     checkpoints, and every script end to end
```

## References

- Fedus, Zoph, Shazeer. [Switch Transformers](https://arxiv.org/abs/2101.03961). 2021.
- Lepikhin et al. [GShard](https://arxiv.org/abs/2006.16668). 2020.
- Dai et al. [DeepSeekMoE](https://arxiv.org/abs/2401.06066). 2024.
- Wang et al. [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664). 2024.
- DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). 2024.
- Zhou et al. [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368). 2022.
- Eldan, Li. [TinyStories](https://arxiv.org/abs/2305.07759). 2023.
- Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). 2021.
- Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). 2020.

## License

MIT
