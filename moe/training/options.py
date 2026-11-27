"""The scripts/train.py command-line flags. The README and the run scripts rely on their names and defaults."""
import argparse


def parse_args(description: str | None = None) -> argparse.Namespace:
    """Parses sys.argv. description is the text --help prints above the flags, and train.py passes its docstring."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    run = parser.add_argument_group("run")
    run.add_argument("--name", default=None, help="run name (default: derived from ffn/router)")
    run.add_argument("--out_dir", default="runs")
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--device", default="auto")
    run.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default="tinystories", choices=["shakespeare", "tinystories"])
    data.add_argument("--tokenizer", default="byte", choices=["byte", "gpt2"])
    data.add_argument("--max_mb", type=int, default=200, help="MB of TinyStories text to use")
    data.add_argument("--data_dir", default=None,
                      help="use an already-tokenized dataset dir (meta.json, train.bin, val.bin) "
                           "instead of --dataset")

    model = parser.add_argument_group("model")
    model.add_argument("--ffn", default="moe", choices=["moe", "dense"])
    model.add_argument("--router", default="aux_loss", choices=["aux_loss", "aux_free", "expert_choice"])
    model.add_argument("--impl", default="triton", choices=["triton", "naive"])
    model.add_argument("--seq_len", type=int, default=256)
    model.add_argument("--d_model", type=int, default=256)
    model.add_argument("--n_layers", type=int, default=6)
    model.add_argument("--n_heads", type=int, default=8)
    model.add_argument("--n_experts", type=int, default=16)
    model.add_argument("--top_k", type=int, default=4)
    model.add_argument("--expert_hidden", type=int, default=64)
    model.add_argument("--n_shared", type=int, default=2)
    model.add_argument("--first_k_dense", type=int, default=0)
    model.add_argument("--dense_hidden", type=int, default=None, help="default: matched to MoE active params")
    model.add_argument("--aux_loss_coef", type=float, default=0.01)
    model.add_argument("--bias_update_rate", type=float, default=1e-3)
    model.add_argument("--capacity_factor", type=float, default=1.0)
    model.add_argument("--dropout", type=float, default=0.0)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--steps", type=int, default=5000)
    optim.add_argument("--batch_size", type=int, default=64)
    optim.add_argument("--grad_accum", type=int, default=1)
    optim.add_argument("--lr", type=float, default=1e-3)
    optim.add_argument("--min_lr_frac", type=float, default=0.1)
    optim.add_argument("--warmup", type=int, default=200)
    optim.add_argument("--weight_decay", type=float, default=0.1)
    optim.add_argument("--grad_clip", type=float, default=1.0)

    log = parser.add_argument_group("logging")
    log.add_argument("--eval_interval", type=int, default=250)
    log.add_argument("--eval_iters", type=int, default=50)
    log.add_argument("--log_interval", type=int, default=25)
    return parser.parse_args()
