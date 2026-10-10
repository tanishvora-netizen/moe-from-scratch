"""
Custom Triton grouped-GEMM kernels for the MoE expert computation.

After routing, every (token, expert) assignment i needs

    C[i] = A[row_idx[i]] @ W[expert(i)]

Assignments are sorted by expert, so expert e owns the contiguous row slice
[offsets[e], offsets[e+1]). The naive implementation loops over experts in
Python: a gather + 2-3 small GEMMs per expert = dozens of kernel launches per
layer, most of them tiny. Here ALL experts run in ONE launch:

  * each program computes one BLOCK_M x BLOCK_N output tile,
  * tiles never straddle experts (a precomputed "tile schedule" maps each
    tile -> (expert, row range)), so a tile only touches one weight matrix,
  * the gather A[row_idx] is fused into the kernel's loads, so the permuted
    copy of the activations is never materialized.

Backward (implemented below, so the kernel is used for training too):
  dA[row_idx[i]] += dC[i] @ W[e]^T       -> same forward kernel, W read transposed via strides
  dW[e] = sum_{i in e} A[row_idx[i]]^T dC[i]  -> a separate weight-gradient kernel
"""
import os

import torch
import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"  # CPU interpreter mode (for tests)

# Tile height along the token/row dimension. The tile schedule depends on it,
# so it is fixed instead of autotuned (BLOCK_N / BLOCK_K / warps are autotuned).
BLOCK_M = 64


# ----------------------------------------------------------------------------------------------
# Tile schedule (pure torch, no host sync)
# ----------------------------------------------------------------------------------------------
def make_tile_schedule(offsets: torch.Tensor, M: int, block_m: int = BLOCK_M):
    """For each M-tile, return (expert id, first row, end row).

    The number of real tiles depends on the routing, which lives on the GPU. Instead
    of syncing to read it, we launch an upper bound of cdiv(M, block_m) + E tiles;
    surplus tiles get start == end and exit immediately inside the kernel.
    """
    E = offsets.numel() - 1
    counts = offsets[1:] - offsets[:-1]
    tiles = (counts + block_m - 1) // block_m              # tiles per expert
    tile_end = torch.cumsum(tiles, 0)                      # inclusive prefix sum
    max_tiles = triton.cdiv(M, block_m) + E
    t = torch.arange(max_tiles, device=offsets.device, dtype=tile_end.dtype)
    expert = torch.searchsorted(tile_end, t, right=True)   # first expert whose tile range ends after t
    valid = expert < E
    expert = expert.clamp(max=E - 1)
    local = t - (tile_end - tiles)[expert]                 # tile index within its expert
    start = offsets[expert] + local * block_m
    end = offsets[expert + 1]
    start = torch.where(valid, start, end)                 # invalid tiles -> empty range
    return expert.to(torch.int32), start.to(torch.int32), end.to(torch.int32)


# ----------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------
@triton.jit
def _grouped_gemm_fwd_kernel(
    A, B, C, row_idx_ptr,
    tile_expert_ptr, tile_start_ptr, tile_end_ptr,
    N, K,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_ROW_IDX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[rows of tile] = A[row_idx[rows]] @ B[expert]   (one BLOCK_M x BLOCK_N tile per program)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = tl.load(tile_start_ptr + pid_m)
    row_end = tl.load(tile_end_ptr + pid_m)
    if row_start >= row_end:      # surplus tile from the upper-bound launch
        return
    expert = tl.load(tile_expert_ptr + pid_m).to(tl.int64)

    offs_m = row_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < row_end
    if HAS_ROW_IDX:               # fused gather: read token rows straight from un-permuted x
        a_rows = tl.load(row_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
    else:
        a_rows = offs_m.to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + a_rows[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _grouped_gemm_wgrad_kernel(
    A, DC, DB, row_idx_ptr, offsets_ptr,
    N, K,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    stride_be, stride_bk, stride_bn,
    HAS_ROW_IDX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """dB[e, k-tile, n-tile] = sum over expert e's rows of A[row_idx]^T @ dC."""
    expert = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)

    start = tl.load(offsets_ptr + expert).to(tl.int32)
    end = tl.load(offsets_ptr + expert + 1).to(tl.int32)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in range(start, end, BLOCK_M):   # experts with 0 tokens skip the loop -> dB = 0
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end
        if HAS_ROW_IDX:
            a_rows = tl.load(row_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
        else:
            a_rows = offs_m.to(tl.int64)
        a = tl.load(A + a_rows[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)            # (BM, BK)
        g = tl.load(DC + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn,
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)            # (BM, BN)
        acc += tl.dot(tl.trans(a), g)

    db_ptrs = DB + expert.to(tl.int64) * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    tl.store(db_ptrs, acc.to(DB.dtype.element_ty), mask=mask_k[:, None] & mask_n[None, :])


# Autotune on a real GPU; fixed configs in interpreter mode (autotuning needs CUDA timers).
if INTERPRET:
    _fwd_kernel, _FWD_FIXED = _grouped_gemm_fwd_kernel, dict(BLOCK_N=32, BLOCK_K=32)
    _wgrad_kernel, _WGRAD_FIXED = _grouped_gemm_wgrad_kernel, dict(BLOCK_M=32, BLOCK_K=32, BLOCK_N=32)
else:
    _fwd_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
            for bn, bk, w, s in [(64, 32, 4, 3), (64, 64, 4, 3), (128, 32, 4, 3),
                                 (128, 64, 8, 3), (64, 32, 2, 4), (128, 32, 4, 4)]
        ],
        key=["N", "K"],
    )(_grouped_gemm_fwd_kernel)
    _wgrad_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": bm, "BLOCK_K": bk, "BLOCK_N": bn}, num_warps=w, num_stages=s)
            for bm, bk, bn, w, s in [(32, 64, 64, 4, 3), (64, 64, 64, 4, 3), (64, 128, 64, 8, 3),
                                     (64, 64, 128, 8, 3), (32, 128, 128, 8, 3)]
        ],
        key=["N", "K"],
    )(_grouped_gemm_wgrad_kernel)
    _FWD_FIXED, _WGRAD_FIXED = {}, {}


# ----------------------------------------------------------------------------------------------
# Launchers
# ----------------------------------------------------------------------------------------------
def _launch_fwd(a, b, sched, M, row_idx=None):
    """C (M, N) = A[row_idx] @ B[expert]. b may be a strided view (used for B^T in backward)."""
    tile_expert, tile_start, tile_end = sched
    K = a.shape[1]
    E, Kb, N = b.shape
    assert Kb == K, f"inner dims mismatch: {Kb} vs {K}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0:
        return c
    n_tiles = tile_expert.numel()
    grid = lambda meta: (n_tiles, triton.cdiv(N, meta["BLOCK_N"]))
    _fwd_kernel[grid](
        a, b, c, row_idx if row_idx is not None else tile_start,
        tile_expert, tile_start, tile_end,
        N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        HAS_ROW_IDX=row_idx is not None,
        BLOCK_M=BLOCK_M,
        **_FWD_FIXED,
    )
    return c


def _launch_wgrad(a, dc, offsets, row_idx, E):
    """dB (E, K, N) in fp32."""
    K, N = a.shape[1], dc.shape[1]
    db = torch.empty((E, K, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (E, triton.cdiv(K, meta["BLOCK_K"]), triton.cdiv(N, meta["BLOCK_N"]))
    _wgrad_kernel[grid](
        a, dc, db, row_idx if row_idx is not None else offsets, offsets,
        N, K,
        a.stride(0), a.stride(1),
        dc.stride(0), dc.stride(1),
        db.stride(0), db.stride(1), db.stride(2),
        HAS_ROW_IDX=row_idx is not None,
        **_WGRAD_FIXED,
    )
    return db


def _compute_dtype(t: torch.Tensor) -> torch.dtype:
    """Respect torch.autocast: run the kernel in bf16/fp16 when autocast is on."""
    if t.is_cuda and torch.is_autocast_enabled():
        try:
            return torch.get_autocast_dtype("cuda")
        except AttributeError:  # torch < 2.4
            return torch.get_autocast_gpu_dtype()
    return t.dtype


# ----------------------------------------------------------------------------------------------
# Autograd wrapper
# ----------------------------------------------------------------------------------------------
class GroupedGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offsets, row_idx):
        cdt = _compute_dtype(a)
        a_c = a.to(cdt).contiguous()
        b_c = b.to(cdt).contiguous()
        M = row_idx.numel() if row_idx is not None else a_c.shape[0]
        sched = make_tile_schedule(offsets, M)
        c = _launch_fwd(a_c, b_c, sched, M, row_idx)

        ctx.save_for_backward(a_c, b_c, offsets, row_idx)
        ctx.sched, ctx.M = sched, M
        ctx.a_dtype, ctx.b_dtype = a.dtype, b.dtype
        return c

    @staticmethod
    def backward(ctx, dc):
        a_c, b_c, offsets, row_idx = ctx.saved_tensors
        dc = dc.to(a_c.dtype).contiguous()
        da = db = None
        if ctx.needs_input_grad[0]:
            # dA_rows = dC @ B[e]^T : reuse the forward kernel with B viewed as (E, N, K)
            da_rows = _launch_fwd(dc, b_c.transpose(1, 2), ctx.sched, ctx.M, None)
            if row_idx is not None:   # undo the fused gather: scatter-add rows back to tokens
                da = torch.zeros(a_c.shape, device=dc.device, dtype=torch.float32)
                da.index_add_(0, row_idx, da_rows.float())
            else:
                da = da_rows
            da = da.to(ctx.a_dtype)
        if ctx.needs_input_grad[1]:
            db = _launch_wgrad(a_c, dc, offsets, row_idx, b_c.shape[0]).to(ctx.b_dtype)
        return da, db, None, None


def grouped_gemm(a, b, offsets, row_idx=None):
    """
    a:       (T, K) activations (rows gathered via row_idx if given, else used as-is)
    b:       (E, K, N) stacked expert weights
    offsets: (E+1,) int64, expert e owns output rows [offsets[e], offsets[e+1])
    row_idx: (M,) int64 or None, source row of `a` for every output row
    returns  (M, N)
    """
    return GroupedGEMM.apply(a, b, offsets, row_idx)
