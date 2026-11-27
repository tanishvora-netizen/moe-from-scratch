"""ModelConfig holds every model hyperparameter and checks them when it is created.
The model, the routers and checkpoint files all read it.
"""
from dataclasses import dataclass
from typing import Literal, get_args

FfnType = Literal["moe", "dense"]
RouterName = Literal["aux_loss", "aux_free", "expert_choice"]
MoEImpl = Literal["triton", "naive"]


def _check_choice(name, value, options):
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


@dataclass
class ModelConfig:
    vocab_size: int = 256            # 256 fits the byte-level tokenizer
    seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0
    rope_base: float = 10000.0

    ffn_type: FfnType = "moe"
    dense_hidden: int | None = None  # dense FFN hidden size, None means matched_dense_hidden

    n_routed_experts: int = 16
    expert_hidden: int = 64          # hidden size of each routed and each shared expert
    top_k: int = 4                   # routed experts per token
    n_shared_experts: int = 2        # experts that every token goes through
    first_k_dense: int = 0           # layers before this index use a dense FFN. DeepSeek models use at least 1.

    router: RouterName = "aux_loss"
    aux_loss_coef: float = 0.01      # balance loss weight, read only by the "aux_loss" router
    bias_update_rate: float = 1e-3   # bias step gamma, read only by the "aux_free" router
    capacity_factor: float = 1.0     # scales each expert's token capacity, read only by the "expert_choice" router

    moe_impl: MoEImpl = "triton"     # "triton" runs the grouped-GEMM kernel, "naive" loops over experts in Python

    def __post_init__(self):
        # Raise, not assert, so the checks still run under `python -O`.
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError(f"head_dim ({self.d_model // self.n_heads}) must be even for RoPE")
        _check_choice("ffn_type", self.ffn_type, get_args(FfnType))
        _check_choice("router", self.router, get_args(RouterName))
        _check_choice("moe_impl", self.moe_impl, get_args(MoEImpl))
        if not 1 <= self.top_k <= self.n_routed_experts:
            raise ValueError(f"top_k ({self.top_k}) must be between 1 and n_routed_experts ({self.n_routed_experts})")
        if self.dense_hidden is None:
            self.dense_hidden = self.matched_dense_hidden

    @property
    def matched_dense_hidden(self) -> int:
        """Dense FFN hidden size with the same active params as one MoE layer."""
        return (self.n_shared_experts + self.top_k) * self.expert_hidden
