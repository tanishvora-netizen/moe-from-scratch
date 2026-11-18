from dataclasses import dataclass


def _check_choice(name, value, options):
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


@dataclass
class ModelConfig:
    # ---- core transformer ----
    vocab_size: int = 256            # 256 = byte-level tokenizer
    seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0
    rope_base: float = 10000.0

    # ---- feed-forward ----
    ffn_type: str = "moe"            # "moe" | "dense"
    dense_hidden: int = 384          # dense FFN hidden size. 384 = (n_shared + top_k) * expert_hidden,
                                     # i.e. the dense baseline matches the MoE's *active* FFN params.

    # ---- DeepSeek-MoE style experts ----
    n_routed_experts: int = 16       # fine-grained routed experts
    expert_hidden: int = 64          # hidden size of each routed expert
    top_k: int = 4                   # routed experts per token
    n_shared_experts: int = 2        # always-on experts, merged into one SwiGLU of size n_shared*expert_hidden
    first_k_dense: int = 0           # first k layers use a dense FFN instead of MoE (DeepSeek uses >= 1)

    # ---- routing / load balancing ----
    router: str = "aux_loss"         # "aux_loss" | "aux_free" | "expert_choice"
    aux_loss_coef: float = 0.01      # Switch/GShard balance loss weight      (router="aux_loss")
    bias_update_rate: float = 1e-3   # gamma for DeepSeek-V3 bias balancing  (router="aux_free")
    capacity_factor: float = 1.0     # capacity = cf * tokens * top_k / E    (router="expert_choice")

    # ---- implementation ----
    moe_impl: str = "triton"         # "triton" (custom grouped-GEMM kernel) | "naive" (python loop over experts)

    def __post_init__(self):
        # Raise instead of assert so the checks still run under `python -O`.
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError(f"head_dim ({self.d_model // self.n_heads}) must be even for RoPE")
        _check_choice("ffn_type", self.ffn_type, ("moe", "dense"))
        _check_choice("router", self.router, ("aux_loss", "aux_free", "expert_choice"))
        _check_choice("moe_impl", self.moe_impl, ("triton", "naive"))
        if not 1 <= self.top_k <= self.n_routed_experts:
            raise ValueError(f"top_k ({self.top_k}) must be between 1 and n_routed_experts ({self.n_routed_experts})")

    @property
    def matched_dense_hidden(self) -> int:
        """Dense FFN hidden size with the same active params as one MoE layer."""
        return (self.n_shared_experts + self.top_k) * self.expert_hidden
