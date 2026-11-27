"""grouped_gemm runs every expert's matmul C = A[row_idx] @ B[expert] in one Triton launch, with a backward pass.
A holds the activations, B the stacked expert weights and C one row per assignment.
Shapes: T tokens, E experts, M rows of C, K the inner dimension, N the output width.
"""
import torch

from .tile_schedule import make_tile_schedule
from .triton_kernels import launch_forward, launch_weight_grad


def _compute_dtype(t: torch.Tensor) -> torch.dtype:
    """Returns the CUDA autocast dtype when autocast is on, else t.dtype.

    Autocast does not cast the inputs of a custom autograd.Function, so GroupedGEMM looks the dtype up here.
    """
    if t.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return t.dtype


class GroupedGEMM(torch.autograd.Function):
    """Autograd for C = A[row_idx] @ B[expert]. The backward pass is:

        dA[row_idx[i]] += dC[i] @ B[e]^T             the forward kernel again, reading B transposed through strides
        dB[e] = sum_{i in e} A[row_idx[i]]^T dC[i]   a separate weight-gradient kernel
    """

    @staticmethod
    def forward(ctx, a, b, offsets, row_idx):
        dtype = _compute_dtype(a)
        a_cast = a.to(dtype).contiguous()
        b_cast = b.to(dtype).contiguous()
        M = row_idx.numel() if row_idx is not None else a_cast.shape[0]
        schedule = make_tile_schedule(offsets, M)
        c = launch_forward(a_cast, b_cast, schedule, M, row_idx)

        ctx.save_for_backward(a_cast, b_cast, offsets, row_idx)
        ctx.schedule, ctx.M = schedule, M
        ctx.a_dtype, ctx.b_dtype = a.dtype, b.dtype
        return c

    @staticmethod
    def backward(ctx, dc):
        a_cast, b_cast, offsets, row_idx = ctx.saved_tensors
        dc = dc.to(a_cast.dtype).contiguous()
        da = db = None
        if ctx.needs_input_grad[0]:
            # dC @ B[e]^T is the forward kernel with B viewed as (E, N, K).
            da_rows = launch_forward(dc, b_cast.transpose(1, 2), ctx.schedule, ctx.M, None)
            if row_idx is not None:   # forward gathered rows of A, so scatter-add row gradients to their tokens
                da = torch.zeros(a_cast.shape, device=dc.device, dtype=torch.float32)
                da.index_add_(0, row_idx, da_rows.float())
            else:
                da = da_rows
            da = da.to(ctx.a_dtype)
        if ctx.needs_input_grad[1]:
            db = launch_weight_grad(a_cast, dc, offsets, row_idx, b_cast.shape[0]).to(ctx.b_dtype)
        return da, db, None, None


def grouped_gemm(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, row_idx: torch.Tensor | None = None
) -> torch.Tensor:
    """Returns C with C[i] = a[row_idx[i]] @ b[e] for every row i that expert e owns, all experts in one launch.

    Rows are grouped by expert, so each expert owns one contiguous slice of C.

    a:       (T, K) activations, read through row_idx when it is given
    b:       (E, K, N) stacked expert weights
    offsets: (E+1,) int64, expert e owns output rows [offsets[e], offsets[e+1])
    row_idx: (M,) int64 or None, the row of a for each output row
    returns  (M, N)
    """
    return GroupedGEMM.apply(a, b, offsets, row_idx)
