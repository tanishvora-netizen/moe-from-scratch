"""The scripts/train.py command-line flags. The README and the run scripts rely on their names and defaults."""
import argparse


def parse_args(description: str | None = None) -> argparse.Namespace:
    """Parses sys.argv. description is the text --help prints above the flags; train.py passes its docstring."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    # run
    parser.add_argument("--name", default=None, help="run name (default: derived from ffn/router)")
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    # data
    parser.add_argument("--dataset", default="tinystories", choices=["shakespeare", "tinystories"])
    parser.add_argument("--tokenizer", default="byte", choices=["byte", "gpt2"])
    parser.add_argument("--max_mb", type=int, default=200, help="MB of TinyStories text to use")
    parser.add_argument("--data_dir", default=None,
                        help="use an already-tokenized dataset dir (meta.json, train.bin, val.bin) "
                             "instead of --dataset")
    # model
    parser.add_argument("--ffn", default="moe", choices=["moe", "dense"])
    parser.add_argument("--router", default="aux_loss", choices=["aux_loss", "aux_free", "expert_choice"])
    parser.add_argument("--impl", default="triton", choices=["triton", "naive"])
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_experts", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--expert_hidden", type=int, default=64)
    parser.add_argument("--n_shared", type=int, default=2)
    parser.add_argument("--first_k_dense", type=int, default=0)
    parser.add_argument("--dense_hidden", type=int, default=None, help="default: matched to MoE active params")
    parser.add_argument("--aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--bias_update_rate", type=float, default=1e-3)
    parser.add_argument("--capacity_factor", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    # optimization
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr_frac", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # logging
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=25)
    return parser.parse_args()
