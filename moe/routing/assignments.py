"""The Routing layout every router returns and the expert kernels consume, and the token-choice builder for it.

Shapes: T tokens, E routed experts, k experts per token, M (token, expert) assignments.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Routing:
    """Which tokens go to which experts, in the layout the expert kernels want.

    Every (token, expert) pair is one assignment, so top-k routing makes M = T * k of them.
    Assignments are sorted by expert, so each expert's rows are one contiguous slice.
    For example, with 3 tokens, 2 experts and k = 1, where tokens 0 and 2 go to expert 1 and token 1 to expert 0:

        sorted_token  = [1, 0, 2]
        sorted_expert = [0, 1, 1]
        offsets       = [0, 1, 3]     # expert 0 owns rows [0, 1), expert 1 owns rows [1, 3)
    """

    sorted_token: torch.Tensor   # (M,)  token index of each assignment, grouped by expert
    sorted_expert: torch.Tensor  # (M,)  expert id of each assignment, non-decreasing
    sorted_weight: torch.Tensor  # (M,)  gate weight, differentiable
    offsets: torch.Tensor        # (E+1,) expert e owns rows [offsets[e], offsets[e+1])
    counts: torch.Tensor         # (E,)  assignments per expert
    aux_loss: torch.Tensor       # scalar, zero unless the router is AuxLossRouter
    n_tokens: int                # T
    # Expert choice can leave a token with no routed expert, and then only the shared experts process it.
    can_drop_tokens: bool = False


def token_choice_routing(top_experts: torch.Tensor, top_weights: torch.Tensor, n_experts: int) -> Routing:
    """Builds the sorted layout from each token's own top-k pick.

    top_experts is (T, k) expert ids. top_weights is (T, k) unnormalized gate weights, which this renormalizes
    to sum to 1 per token. aux_loss is zero, and AuxLossRouter fills it in.
    """
    T, k = top_experts.shape
    top_weights = top_weights / (top_weights.sum(-1, keepdim=True) + 1e-9)
    flat = top_experts.reshape(-1)                                  # (T*k,)
    counts = torch.bincount(flat, minlength=n_experts)
    sorted_expert, order = torch.sort(flat, stable=True)
    return Routing(
        sorted_token=order // k,   # flat position i belongs to token i // k
        sorted_expert=sorted_expert,
        sorted_weight=top_weights.reshape(-1)[order],
        offsets=F.pad(torch.cumsum(counts, 0), (1, 0)),
        counts=counts,
        aux_loss=top_weights.new_zeros(()),
        n_tokens=T,
    )
